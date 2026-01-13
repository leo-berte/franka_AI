#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from custom_msgs.msg import GripperWidth

from cv_bridge import CvBridge
from collections import deque
import threading
import torch
import numpy as np
from math import ceil
import time

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.utils.robotics_math import *
from franka_ai.dataset.utils import get_configs_dataset, build_delta_timestamps
from franka_ai.inference.utils import get_configs_inference
from franka_ai.models.factory import get_policy_class
from franka_ai.models.utils import get_configs_models


# TODO:

# in ros2_webcam node abilitare header!
# capire perche gripper state e action non stanno a 30hz nei synced deltas

# 0) Come essere sicuri che codice funziona? Plotto su rqt_plot output policy Vs rosbag actions (GT) ? --> registro 1 episodio intero e traino su quello

# 0) A cosa serve action pad? Serve per ACT dentro inference? vedi trasnformrs.py
# 1) traj stitching


"""
Run: ros2 run franka_ai_inference inference_node --ros-args -p use_sim_time:=true
Play rosbag: ros2 bag play bag1.db3 --clock
Run rqt_plot: ros2 run plotjuggler plotjuggler
"""


## Set relative path to inference.yaml before running the node ##
checkpoint_rel_path = "../workspace/outputs/checkpoints/one_bag_act_Test_B/one_bag_act_2026-01-07_18-28-18"


class FrankaInference(Node):

    """
    ROS2 Node for performing real-time inference on a Franka robot using a pre-trained policy.

    This node subscribes to multiple sensor streams, and tt maintains internal buffers for sensor data 
    and builds synchronized snapshots of the robot state to feed into a learned policy. 
    The policy outputs actions for both the end-effector pose and the gripper.

    Features:
        - Real-time buffering and synchronization of sensor data and past actions according to 'fps_sampling_hist' 
        - Multi-threaded execution using ReentrantCallbackGroup for subscribers and a
          MutuallyExclusiveCallbackGroup for inference timer to avoid blocking callbacks
        - Converts sensor inputs into tensor-based observations compatible with the trained policy
        - Automatically decompose policy output format (based on dataset.yaml) in the format required by the robot controller
        - Publishes actions to the robot in ROS2 topics

    Workflow:
        1. Subscribers append new sensor messages into corresponding buffers.
        2. Inference timer triggers at `fps_sampling_chunk` Hz.
        3. On timer callback:
            a. Checks if buffers are ready.
            b. Copies and synchronizes buffers to match reference timestamps (according to 'fps_sampling_hist').
            c. Builds tensor-based observations for policy.
            d. Runs policy inference to get actions.
            e. Decompose policy output in the expected robot controller format.
            f. Optionally smooths actions.
            g. Publishes actions to robot control topics.
    """


    def __init__(self):

        super().__init__('FrankaInference')

        # Get configs about inference
        inference_cfg = get_configs_inference(f"{checkpoint_rel_path}/inference.yaml")

        # Get parameter values from inference.yaml  
        policy_name = inference_cfg["policy_name"]
        offline_test = inference_cfg["offline_test"]
        self.fps_dataset = inference_cfg["fps_dataset"] 
        self.alpha = inference_cfg["output_filter_alpha"]
        self.smooth_output = inference_cfg["smooth_output"]

        # Get configs about dataset, training related to the saved checkpoint
        dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset(f"{checkpoint_rel_path}/dataset.yaml")
        models_cfg = get_configs_models(f"{checkpoint_rel_path}/models.yaml")
        model_cfg = models_cfg[policy_name]

        # Get parameter values
        self.device = dataloader_cfg["device"]
        self.N_history = model_cfg["params"].get("n_obs_steps") or model_cfg["params"].get("N_history")
        self.N_chunk = model_cfg["params"].get("horizon") or model_cfg["params"].get("chunk_size") or model_cfg["params"].get("N_chunk")
        self.n_action_steps = model_cfg["params"].get("n_action_steps")
        self.fps_sampling_hist = model_cfg["sampling"]["fps_sampling_hist"]
        self.fps_sampling_chunk = model_cfg["sampling"]["fps_sampling_chunk"]
        self.orientation_type = transforms_cfg["orientations"]["type"]
        self.include_actions = transforms_cfg["action"]["include"]
        self.state_ranges = dataset_cfg["state_slices"]
        self.state_slices = {k: slice(v[0], v[1]) for k, v in self.state_ranges.items()}

        # Consistency checks
        if self.fps_sampling_hist != self.fps_sampling_chunk:
            raise ValueError("fps_sampling_hist must be the same as fps_sampling_chunk.")
    
        # Prepare transforms for inference
        self.tf_inference = CustomTransforms(
            dataset_cfg=dataset_cfg,
            transforms_cfg=transforms_cfg,
            model_cfg=model_cfg,
            train=False
        )

        # Setup cv2 bridge
        self.bridge = CvBridge()
        
        # Protect shared data from concurrent access
        self.buffer_lock = threading.Lock()

        # Init buffers
        
        fastest_freq_from_sensors = 90 # joint states frequency [Hz]
        deque_max_len = int(self.N_history * ceil(fastest_freq_from_sensors/self.fps_sampling_hist) * 1.5) # Add 1.5 as additional 50% margin

        self.buffers = {
            "webcam1": deque(maxlen=deque_max_len),         # (H,W,C)
            "webcam2": deque(maxlen=deque_max_len),         # (H,W,C)
            "webcam3": deque(maxlen=deque_max_len),         # (H,W,C)
            "realsense_rgb": deque(maxlen=deque_max_len),   # (H,W,C)
            "realsense_depth": deque(maxlen=deque_max_len), # (H,W)
            "q": deque(maxlen=deque_max_len),
            "qdot": deque(maxlen=deque_max_len),
            "tau": deque(maxlen=deque_max_len),
            "fext": deque(maxlen=deque_max_len),
            "cart_pos_curr": deque(maxlen=deque_max_len),
            "cart_quat_curr": deque(maxlen=deque_max_len),
            "gripper_state": deque(maxlen=deque_max_len),
            "action": deque(maxlen=deque_max_len),
        }

        # Params
        self.p_base = torch.zeros(3) # init
        self.R_base = torch.eye(3)   # init

        # Enable MultiThreadedExecutor + ReentrantCallbacks to enbale callbacks/timer to be processed without delays
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.sub_group = ReentrantCallbackGroup()

        # Subscribers
   
        self.webcam1_sub = self.create_subscription(CompressedImage, '/webcam1/image_raw/compressed', self.webcam1_callback, 10, callback_group=self.sub_group)
        self.webcam2_sub = self.create_subscription(CompressedImage, '/webcam2/image_raw/compressed', self.webcam2_callback, 10, callback_group=self.sub_group)
        self.webcam3_sub = self.create_subscription(CompressedImage, '/webcam3/image_raw/compressed', self.webcam3_callback, 10, callback_group=self.sub_group)
        self.realsense_rgb_sub = self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.realsense_rgb_callback, 10, callback_group=self.sub_group)
        self.realsense_depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.realsense_depth_callback, 10, callback_group=self.sub_group)
        self.joint_state_sub = self.create_subscription(JointState, '/cartesian_impedance/joint_state', self.joint_state_callback, 10, callback_group=self.sub_group)
        self.gripper_state_sub = self.create_subscription(GripperWidth, '/panda_gripper/width', self.gripper_state_callback, 10, callback_group=self.sub_group)
        self.fext_sub = self.create_subscription(WrenchStamped, '/cartesian_impedance/f_ext_cart', self.fext_callback, 10, callback_group=self.sub_group)
        self.cart_pose_curr_sub = self.create_subscription(PoseStamped, '/cartesian_impedance/cartesian_pos_curr', self.cart_pose_curr_callback, 10, callback_group=self.sub_group)
        
        # Publisher

        cart_pose_action_pub_topic = '/cartesian_impedance/equilibrium_pose' if offline_test == False else '/cartesian_impedance/equilibrium_pose_offline_test'
        gripper_action_pub_topic = '/panda_gripper/gripper_command' if offline_test == False else '/panda_gripper/gripper_command_offline_test'

        self.cart_pose_action_pub = self.create_publisher(PoseStamped, cart_pose_action_pub_topic, 10)
        self.gripper_action_pub = self.create_publisher(GripperWidth, gripper_action_pub_topic, 10)

        # Timer according to framerate
        self.timer = self.create_timer(1.0 / self.fps_sampling_chunk, self.inference_timer, callback_group=self.timer_group)

        # Load policy
        PolicyClass = get_policy_class(policy_name)
        self.policy = PolicyClass.from_pretrained(f"{checkpoint_rel_path}/best_model.pt")
        self.policy.reset() # reset the policy to prepare for rollout

        # Compute policy steps
        self.current_step = 0

        # Build the delta_timestamps dict for history and future        
        self.delta_timestamps = build_delta_timestamps(dataset_cfg["features"], 
                                                       self.N_history, 
                                                       self.N_chunk, 
                                                       self.fps_dataset, # dataset_meta.fps, 
                                                       self.fps_sampling_hist, 
                                                       self.fps_sampling_chunk)

        self.get_logger().info("FrankaInference node initialized successfully")

    def webcam1_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        with self.buffer_lock:
            self.buffers["webcam1"].append((msg.header.stamp, rgb))

        # print("webcam1 header.stamp:", msg.header.stamp)

    def webcam2_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        with self.buffer_lock:
            self.buffers["webcam2"].append((msg.header.stamp, rgb))

    def webcam3_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        with self.buffer_lock:
            self.buffers["webcam3"].append((msg.header.stamp, rgb))

    def realsense_rgb_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        with self.buffer_lock:
            self.buffers["realsense_rgb"].append((msg.header.stamp, rgb))

    def realsense_depth_callback(self, msg):

        # Depth is a 16-bit uint image; convert to float32 meters
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  
        # depth = depth.astype('float32')

        # # Replace invalid values (0 or NaN) with last valid or 0 + normalize ??
        # depth[depth <= 0] = 0.0

        with self.buffer_lock:
            self.buffers["realsense_depth"].append((msg.header.stamp, depth))
    
    def joint_state_callback(self, msg):

        # Joint positions, velocities, efforts
        with self.buffer_lock:
            self.buffers["q"].append((msg.header.stamp, list(msg.position)))
            self.buffers["qdot"].append((msg.header.stamp, list(msg.velocity)))
            self.buffers["tau"].append((msg.header.stamp, list(msg.effort)))

    def gripper_state_callback(self, msg):
        
        # Gripper width
        with self.buffer_lock:
            self.buffers["gripper_state"].append((msg.header.stamp, msg.width))

    def fext_callback(self, msg):

        # Extract gripper forces
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        tx = msg.wrench.torque.x
        ty = msg.wrench.torque.y
        tz = msg.wrench.torque.z

        with self.buffer_lock:
            self.buffers["fext"].append((msg.header.stamp, [fx, fy, fz, tx, ty, tz]))

    def cart_pose_curr_callback(self, msg):

        # Cart pose
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        with self.buffer_lock:
            self.buffers["cart_pos_curr"].append((msg.header.stamp, [px, py, pz]))
            self.buffers["cart_quat_curr"].append((msg.header.stamp, [qx, qy, qz, qw]))

    def ready(self):

        """
        Check that all sensor buffers have enough data to perform inference.
        If the action buffer is empty but sufficient state data exists (cartesian
        position, orientation and gripper state), it initializes the action buffer with the
        current robot state.

        Returns:
            bool: True if all required buffers have data, False otherwise.
        """

        # Check all the state buffers have at least 1 element
        for k, buf in self.buffers.items():
            if (k!="action" and len(buf)<=0):
                return False
            
        # Eventually initialize action buffer with current robot state
        if len(self.buffers["action"]) <= 0 \
            and len(self.buffers["cart_pos_curr"]) > 0 \
            and len(self.buffers["cart_quat_curr"]) > 0 \
            and len(self.buffers["gripper_state"]) > 0:

            # Get current robot state
            curr_pos = self.buffers["cart_pos_curr"][-1][1]
            curr_quat = self.buffers["cart_quat_curr"][-1][1]
            curr_grip = CustomTransforms.gripper_action_continuous2discrete(self.buffers["gripper_state"][-1][1])
            
            # Compose action and add to buffer
            curr_action = curr_pos + curr_quat + [curr_grip]
            self.buffers["action"].append((self.get_clock().now().to_msg(), curr_action))

        return len(self.buffers["action"]) > 0 

    def get_data_at_ref_times(self, reference_times, buffer):

        """
        Retrieve sub-sampled buffer data aligned to a list of reference timestamps.

        For each reference time, the closest previous timestamp in the buffer is selected
        to create a synchronized list of data points. This ensures that the returned data
        is temporally aligned with the reference timeline, useful for building historical
        observations.

        Args:
            reference_times: List of reference timestamps in seconds.
            buffer: List of (ros_time, value) tuples, sorted by ros_time.

        Returns:
            List: List of (ros_time, value) tuples aligned to the reference times.
        """

        out = []
        idx = 0
        times = [t.sec + t.nanosec * 1e-9 for t, _ in buffer] # get absolute data timestamps
        
        # Align absolute data timestamps to absolute delta timestamps 
        for rt in reference_times: 
            while idx + 1 < len(times) and times[idx + 1] <= rt:
                idx += 1
            out.append((buffer[idx][0], buffer[idx][1]))

        return out

    def get_buffer_synced(self, buffers):

        """
        Align multiple buffers to a common reference timeline based on the current time 
        and precomputed delta timestamps.

        Args:
            buffers: Dictionary of buffers to sync, where each key is a sensor name 
                and each value is a list of (ros_time, value) tuples sorted by timestamp.

        Returns:
            Dict: Dictionary of buffers aligned to the reference times. Each buffer has 
                the same length as `delta_timestamps['observation.state']`.
            
        Example 1
        buffer: [(t0, x0)] 
        buffer (N_hist=3): [(t0, x0), (t0, x0), (t0, x0)] 
        
        Example 2
        buffer: [(t0, x0), (t1, x1), (t2, x2), (t3, x3), (t4, x4), (t5, x5), (t6, x6)] 
        buffer (N_hist=3): [(t0, x0), (t2, x2), (t6, x6)] 
        """

        # Get current time in ROS
        t_curr = self.get_clock().now().to_msg()
        t_curr_sec = t_curr.sec + t_curr.nanosec * 1e-9
        
        # Get relative delta timestamps
        delta_t_state = self.delta_timestamps['observation.state']

        # Compute absolute delta timestamps
        ref_times_state = [t_curr_sec + dt for dt in delta_t_state]  

        print("NON synced deltas: ")
        self.debug_print_deltatimes(buffers)

        buffer_synced = {}

        # Sub-sample buffers data to align with delta_timestamps

        for k, buf in buffers.items():

            buffer_synced[k] = self.get_data_at_ref_times(ref_times_state, buf)

        print("synced deltas: ")
        self.debug_print_deltatimes(buffer_synced)

        return buffer_synced

    def stack_data(self, buff, is_image=False):
        
        """
        Convert a list of buffered data into a batched torch tensor suitable for policy input.

        This method extracts only the values from a buffer of (timestamp, value) tuples, 
        converts them to torch tensors, and stacks them along the history dimension. 
        An additional batch dimension is added as the first dimension (B=1).

        Args:
            buff: List of tuples (ros_time, value) extracted from a buffer.
            is_image: If True, treat values as images and permute from HWC to CHW.

        Returns:
            torch.Tensor: Tensor of shape (B=1, N_history, ...) for scalar/vector data 
                or (B=1, N_history, C, H, W) for images.
        """

        if is_image: # (include here also depth image eventually)
            # Convert to tensors and stack images to: (1, N_history, H, W, C)
            tensor_values = torch.stack([torch.from_numpy(b[1]).pin_memory() for b in buff], dim=0).unsqueeze(0)

            # Move to device
            tensor_values = tensor_values.to(self.device, non_blocking=True)

            # Convert from HWC to CHW + convert from uint8 to float32 + normalize [0, 1]
            tensor_values = tensor_values.permute(0, 1, 4, 2, 3).float() / 255.0
        else:
            # Convert to tensors: (1, N_history, ..)
            tensor_values = torch.stack([torch.as_tensor(b[1], dtype=torch.float32).flatten().pin_memory() for b in buff], dim=0).unsqueeze(0)

            # Move to device
            tensor_values = tensor_values.to(self.device, non_blocking=True)

        return tensor_values

    def build_obs(self):

        """
        Build a synchronized observation dictionary for the policy network.

        This method performs the following steps:
        1. Creates a thread-safe snapshot of all current buffers.
        2. Synchronizes and subsamples each buffer according to precomputed delta timestamps 
        to ensure the correct temporal alignment for history (N_history).
        3. Converts buffered data into torch tensors + move data to device.
        4. Constructs an observation dictionary with:
        - `"observation.state"`: concatenated state tensor
        - `"observation.images.xxx`: synchronized image tensors from multiple cameras
        - `"action"`: past actions for transforms (removed later)
        5. If relative end-effector poses are included, computes the base position and rotation 
        from the last observed state.
        6. Applies custom transforms (`self.tf_inference.transform`) to match the dataset preprocessing.
        7. Convert all tensors from (B, N_history, D) → (B, D) 
        by selecting only the last timestep, as required by the policy 'forward()' format.

        Returns:
            dict: Observation dictionary ready for policy input. Keys include:
                - `"observation.state"`: tensor (B, D)
                - `"observation.images.xxx"`: tensor (B, C, H, W)
        """
        
        # Profile time to sync buffer
        t0 = time.perf_counter()
        
        # Make a snapshot copy inside lock
        with self.buffer_lock:
            buffer_copies = {k: list(v) for k, v in self.buffers.items()}

        # Sub-sample buffers data to align with delta_timestamps --> len == N_history
        buffer_synced = self.get_buffer_synced(buffer_copies)

        # Profile time to sync buffer
        t1 = time.perf_counter()
        print(f"[Build obs] sync: {(t1-t0)*1000:.3f} ms")

        # Profile time to stack buffers
        t0 = time.perf_counter()

        # Convert data to tensors and create history
        data_tensors = {} 
        for k, buf in buffer_synced.items():
            is_image = "webcam" in k or "rgb" in k
            data_tensors[k] = self.stack_data(buf, is_image=is_image) # (B, N_history, ...)

        # Profile time to stack buffers
        t1 = time.perf_counter()
        print(f"[Build obs] stack: {(t1-t0)*1000:.3f} ms")

        # Rebuild state vector (B, N_hist, D) as in original dataset
        state = torch.cat([
            data_tensors["q"],
            data_tensors["qdot"],
            data_tensors["tau"],
            data_tensors["cart_pos_curr"],
            data_tensors["cart_quat_curr"],
            data_tensors["gripper_state"],
            data_tensors["fext"],                 
        ], dim=-1)

        # Create the policy input dictionary as in original dataset
        observation = {
            "observation.state": state,
            "observation.images.front_cam1": data_tensors["webcam1"],
            "observation.images.front_cam2": data_tensors["webcam2"],
            "observation.images.front_cam3": data_tensors["webcam3"],
            "observation.images.gripper_camera": data_tensors["realsense_rgb"],
            "action": data_tensors["action"] # transforms need to include past actions in the state
        }

        # Get base pose (last observed pose) to compute ee_pose_relative --> ee_pose_absolute
        if "ee_pose_relative" in self.include_actions and self.current_step % self.n_action_steps == 0:
            p_base = observation["observation.state"][..., self.state_slices["ee_pos"]][:, -1, :]   # (B,3)
            quat_base = observation["observation.state"][..., self.state_slices["ee_quaternion"]][:, -1, :]  # (B,4)
            quat_base = quat_base[:, [3,0,1,2]]          # xyzw → wxyz
            R_base = quaternion_to_matrix(quat_base)     # (B,3,3)
            self.p_base = p_base.squeeze(0)
            self.R_base = R_base.squeeze(0)
        
        # Profile time for transforms
        t0 = time.perf_counter()
            
        # Apply custom transforms
        observation = self.tf_inference.transform(observation) # in/out: (B, N_h, ...)

        # Profile time for transforms
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[Build obs] transforms: {(t1-t0)*1000:.3f} ms")

        # Remove key "action"
        observation.pop("action", None)

        # Convert (B,N_hist, D) in (B, D) by taking last timestep (input format required by policy)
        for k, v in observation.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 3: 
                v = v[:, -1, ...].contiguous()
                observation[k] = v

        return observation

    def smooth_action(self, new_action, prev_action):
        
        """
        Apply a simple low-pass filter to smooth the robot action.

        This function smooths only the end-effector position and orientation 
        (quaternion) using an exponential moving average. The gripper command is left unchanged.

        Args:
            new_action: The new action predicted by the policy, shape (8,):
                [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]
            prev_action: The previous action applied, same shape as `new_action`.

        Returns:
            np.ndarray: The filtered action, same shape as input (8,). Position and quaternion
                        components are smoothed, quaternion is renormalized to maintain unit norm,
                        gripper command is directly copied from `new_action`.

        """
            
        # Split components

        pos_new  = new_action[:3]
        quat_new = new_action[3:7]
        grip_new = new_action[7] 

        pos_prev  = prev_action[:3]
        quat_prev = prev_action[3:7]

        # Filter position
        pos_f = self.alpha * pos_new + (1 - self.alpha) * pos_prev

        # Filter quaternion
        quat_f = self.alpha * quat_new + (1 - self.alpha) * quat_prev
        quat_f = quat_f / np.linalg.norm(quat_f)   # renormalize

        # Gripper unchanged
        grip_f = grip_new

        # Reassemble
        return np.concatenate([pos_f, quat_f, [grip_f]])

    def inference_timer(self):

        """
        This method is called periodically according to `self.fps_sampling_chunk`. It performs the following steps:

        1. Checks if all required buffers have data (self.ready()).
        2. Builds the observation from buffers (self.build_obs()).
        3. Runs policy inference to compute the next action.
        4. Processes action components (ee_pose_absolute/relative, gripper).
        5. Saves the action in the buffer and publishes it as ROS messages.
        """

        # Wait at least 1 data for each buffer
        if (not(self.ready())):
            return

        # Profile time to build observation
        t0 = time.perf_counter()

        # Build observation
        obs = self.build_obs()

        # Profile time to build observation
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[Timer] build obs: {(t1-t0)*1000:.3f} ms")

        # Profile time for inference
        t0 = time.perf_counter()

        # Inference
        with torch.inference_mode():
            action = self.policy.select_action(obs) # (B, D) --> (B, D)
            # actions = self.policy.diffusion.generate_actions(obs) # (B, N_hist, D) --> (N_chunk, D)
            print("step: ", self.current_step)
            # print("policy action: ", action)

        # Profile time for inference
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[Timer] inference: {(t1-t0)*1000:.3f} ms")

        # Update policy steps
        self.current_step += 1

        # Move to CPU and convert to numpy
        action = action.squeeze(0).to("cpu")

        action_parts = []

        for action_name in self.include_actions:

            if action_name == "ee_pose_absolute":

                pos = action[:3]

                if self.orientation_type == "quaternion":
                    quat = action[3:7]
                elif self.orientation_type == "axis_angle":
                    quat = axis_angle_to_quaternion(action[3:6]) 
                elif self.orientation_type == "6D":
                    quat = matrix_to_quaternion(rotation_6d_to_matrix(action[3:9]))
                
                quat = normalize_quat(quat)
                # quat = standardize_quaternion(quat) # same result with ON or OFF
                quat = quat[..., [1,2,3,0]] # PyTorch3D (w,x,y,z) → Dataset (x,y,z,w) 

                part = torch.cat([pos, quat], dim=-1)

            if action_name == "ee_pose_relative":
                
                # Get relative poses
                pos = action[:3]
                ori_6d = action[3:9]
                R = rotation_6d_to_matrix(ori_6d)
                
                # Transform relative ee_pose in absolute ee_pose wrt last observed state
                part = get_absolute_pose_wrt_last_state(pos, R, self.p_base.to("cpu"), self.R_base.to("cpu")) # quaternion notation for orientation

            elif action_name == "gripper":
                action[-1] = CustomTransforms.gripper_action_continuous2discrete(action[-1]) # Convert gripper in binary {0,1}
                part = action[-1:]

            # Add part to state vector
            action_parts.append(part)

        # Build final action as expected by controller
        action_pre_tf = torch.cat(action_parts, dim=-1).numpy()
        # print("action policy", action_pre_tf)
        
        # Eventually smooth policy output
        if self.smooth_output == True:

            # Get previous action from buffer (take only the action values)
            prev_action = np.array(self.buffers["action"][-1][1], dtype=np.float32)
            # print("prev_action", prev_action)

            # Filter action
            action_pre_tf = self.smooth_action(action_pre_tf, prev_action)
            # print("action policy filtered and applied", action_pre_tf)

        # Save action in buffer safely
        with self.buffer_lock:
            self.buffers["action"].append((self.get_clock().now().to_msg(), action_pre_tf))

        # Set cart pose action
        cart_msg = PoseStamped()
        cart_msg.header.stamp = self.get_clock().now().to_msg()
        cart_msg.pose.position.x = float(action_pre_tf[0])
        cart_msg.pose.position.y = float(action_pre_tf[1])
        cart_msg.pose.position.z = float(action_pre_tf[2])
        cart_msg.pose.orientation.x = float(action_pre_tf[3])
        cart_msg.pose.orientation.y = float(action_pre_tf[4])
        cart_msg.pose.orientation.z = float(action_pre_tf[5])
        cart_msg.pose.orientation.w = float(action_pre_tf[6])

        # Publish
        self.cart_pose_action_pub.publish(cart_msg)

        # Set gripper action
        gripper_msg = GripperWidth()
        gripper_msg.header.stamp = self.get_clock().now().to_msg()
        gripper_msg.width = float(action_pre_tf[-1])

        # Publish
        self.gripper_action_pub.publish(gripper_msg)

    ## HELPERS ##
     
    def debug_print_deltatimes(self, buffers):

        def convert_rostime(ros_time):

            return ros_time.sec + ros_time.nanosec * 1e-9
        
        for k,buff in buffers.items():

            deltas = []

            for i in range(1,len(buff)):

                t_end = convert_rostime(buff[i][0])
                t_start = convert_rostime(buff[i-1][0])
                deltas.append(1/(t_end-t_start+1E-7))
            
            print(f"deltas for {k} [Hz]: ", deltas)


def main(args=None):

    rclpy.init(args=args)
    node = FrankaInference()

    # MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()