#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

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

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.utils import get_configs_dataset, build_delta_timestamps
from franka_ai.inference.utils import get_configs_inference
from franka_ai.models.factory import get_policy_class
from franka_ai.models.utils import get_configs_models


"""
Run: 
Play rosbag: ros2 bag play bag1.db3
"""

# TODO:

# PAST_ACTIONS viene ricreata da output policy.. creare transofrm_inverse per gestirlo senza riscrivere codice per ogni ablation
# --> questo viene fatto per observation perchè ricreo STATE del dataset e transformrs poi lo trattano based on yaml

# 0) build_obs synchronization and feed directly the model observation --> generate_actions(obs) with in:(B, N_hist, D) --> out:(N_chunk, D)
# 1) traj stitching



class FrankaInference(Node):

    def __init__(self):

        super().__init__('FrankaInference')

        # Get configs about inference
        inference_cfg = get_configs_inference("../workspace/configs/config/inference.yaml")

        # Get parameter values from inference.yaml
        self.policy_rate = inference_cfg["policy_rate"]
        self.fps_dataset = inference_cfg["fps_dataset"]
        pretrained_policy_abs_path = inference_cfg["pretrained_policy_abs_path"]
        configs_dataset_rel_path = inference_cfg["configs_dataset_rel_path"]
        configs_models_rel_path = inference_cfg["configs_models_rel_path"]        
        policy_name = inference_cfg["policy_name"]

        # Get configs about dataset, training related to the saved checkpoint
        dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset(configs_dataset_rel_path)
        models_cfg = get_configs_models(configs_models_rel_path)
        model_cfg = models_cfg[policy_name]

        # Get parameter values
        self.device = dataloader_cfg["device"]
        self.N_history = model_cfg["params"].get("n_obs_steps") or model_cfg["params"].get("N_history")
        self.N_chunk = model_cfg["params"].get("horizon") or model_cfg["params"].get("chunk_size") or model_cfg["params"].get("N_chunk")
        self.fps_sampling_hist = model_cfg["sampling"]["fps_sampling_hist"]
        self.fps_sampling_chunk = model_cfg["sampling"]["fps_sampling_chunk"]

        # Prepare transforms for inference
        self.tf_inference = CustomTransforms(
            dataset_cfg=dataset_cfg,
            transforms_cfg=transforms_cfg,
            model_cfg=models_cfg[policy_name],
            train=False
        )

        # Setup cv2 bridge
        self.bridge = CvBridge()
        
        # Protect shared data from concurrent access
        self.buffer_lock = threading.Lock()

        # buffers --> self.buffers = {k: deque(maxlen=self.N_history) for k in self.buffers.keys()}
        self.buffers = {
            "webcam1": deque(maxlen=self.N_history),
            "webcam2": deque(maxlen=self.N_history),
            "webcam3": deque(maxlen=self.N_history),
            "realsense_rgb": deque(maxlen=self.N_history),
            "realsense_depth": deque(maxlen=self.N_history),
            "q": deque(maxlen=self.N_history),
            "qdot": deque(maxlen=self.N_history),
            "tau": deque(maxlen=self.N_history),
            "fext": deque(maxlen=self.N_history),
            "cart_pos_curr": deque(maxlen=self.N_history),
            "cart_quat_curr": deque(maxlen=self.N_history),
            "gripper_state": deque(maxlen=self.N_history),
            "action": deque(maxlen=self.N_history),
            # "cart_pos_filtered_command": deque(maxlen=self.N_history),
            # "cart_ori_filtered_command": deque(maxlen=self.N_history),
            # "gripper_command": deque(maxlen=self.N_history),
        }

        # Params
        self.alpha = 0.05 # filter actions

        # Subscribers
        self.webcam1_sub = self.create_subscription(CompressedImage, '/webcam1/image_raw/compressed', self.webcam1_callback, 10)
        self.webcam2_sub = self.create_subscription(CompressedImage, '/webcam2/image_raw/compressed', self.webcam2_callback, 10)
        self.webcam3_sub = self.create_subscription(CompressedImage, '/webcam3/image_raw/compressed', self.webcam3_callback, 10)
        self.realsense_rgb_sub = self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.realsense_rgb_callback, 10)
        self.realsense_depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.realsense_depth_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/cartesian_impedance/joint_state', self.joint_state_callback, 10)
        self.gripper_state_sub = self.create_subscription(GripperWidth, '/panda_gripper/width', self.gripper_state_callback, 10)
        self.fext_sub = self.create_subscription(WrenchStamped, '/cartesian_impedance/f_ext_cart', self.fext_callback, 10)
        self.cart_pose_curr_sub = self.create_subscription(PoseStamped, '/cartesian_impedance/cartesian_pos_curr', self.cart_pose_curr_callback, 10)
        # self.cart_pose_filtered_action_sub = self.create_subscription(PoseStamped, '/cartesian_impedance/cartesian_pos_des_filt', self.cart_pose_filtered_action_callback, 10)
        # self.gripper_action_sub = self.create_subscription(GripperWidth, '/panda_gripper/gripper_command', self.gripper_action_callback, 10)
        
        # Publisher
        self.cart_pose_action_pub = self.create_publisher(PoseStamped, '/cartesian_impedance/equilibrium_pose', 10)
        self.gripper_action_pub = self.create_publisher(GripperWidth, '/panda_gripper/gripper_command', 10)

        # Timer according to framerate
        self.timer = self.create_timer(1.0 / self.policy_rate, self.inference_timer)

        # Load policy
        PolicyClass = get_policy_class(policy_name)
        self.policy = PolicyClass.from_pretrained(pretrained_policy_abs_path)
        self.policy.reset() # reset the policy to prepare for rollout

        # # Extract input features keys
        # self.policy_input_features_keys = self.policy.config.input_features.keys()
        # self.dataset_input_features_keys = dataset_cfg["features"]["VISUAL"] + dataset_cfg["features"]["STATE"]

        # Build the delta_timestamps dict for history and future        
        self.delta_timestamps = build_delta_timestamps(dataset_cfg["features"], 
                                                       self.N_history, 
                                                       self.N_chunk, 
                                                       self.fps_dataset, # dataset_meta.fps, 
                                                       self.fps_sampling_hist, 
                                                       self.fps_sampling_chunk)

        self.get_logger().info("FrankaInference node initialized successfully")

        ## TEMP ##
        self.step=1

    def webcam1_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        rgb = rgb.astype('float32') / 255.0  # convert to float32 and normalize [0, 1]
        with self.buffer_lock:
            self.buffers["webcam1"].append((msg.header.stamp, rgb))

    def webcam2_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        rgb = rgb.astype('float32') / 255.0  # convert to float32 and normalize [0, 1]
        with self.buffer_lock:
            self.buffers["webcam2"].append((msg.header.stamp, rgb))

    def webcam3_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        rgb = rgb.astype('float32') / 255.0  # convert to float32 and normalize [0, 1]
        with self.buffer_lock:
            self.buffers["webcam3"].append((msg.header.stamp, rgb))

    def realsense_rgb_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        rgb = rgb.astype('float32') / 255.0  # convert to float32 and normalize [0, 1]
        with self.buffer_lock:
            self.buffers["realsense_rgb"].append((msg.header.stamp, rgb))

    def realsense_depth_callback(self, msg):

        # Depth is a 16-bit uint image; convert to float32 meters
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  
        depth = depth.astype('float32')

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
            curr_grip = CustomTransforms.gripper_continuous2discrete(self.buffers["gripper_state"][-1][1])
            
            # Compose action and add to buffer
            curr_action = curr_pos + curr_quat + [curr_grip]
            self.buffers["action"].append((self.get_clock().now().to_msg(), curr_action))

        return len(self.buffers["action"]) > 0 

    def stack_data(self, buffer, is_image=False):

        # [(t0, x0), (t1, x1), (t2, x2)] 
        # [x0, x1, x2, x2, x2] 

        # Get last N_history data + padding
        if len(buffer) < self.N_history:
            n_missing = self.N_history - len(buffer)
            last = buffer[-1] # replicate last item
            items = list(buffer) + [last] * n_missing
        else:
            items = list(buffer)[-self.N_history:]

        # Extract only data (ignore timestamps)
        values = [v for (_, v) in items]

        #  Convert to torch tensors
        if (is_image):
            tensor_values = [torch.from_numpy(v).float() for v in values]
            tensor_values = [v.permute(2, 0, 1).contiguous() for v in tensor_values] # permute images from HWC to CHW
        else:
            tensor_values = []
            for v in values:  
                t = torch.tensor(v, dtype=torch.float32)
                if t.dim() == 0:
                    t = t.unsqueeze(0)  # make scalar into 1D tensor
                tensor_values.append(t)

        # Stack to (N_history, ... ) + add B dimension
        stacked_values = torch.stack(tensor_values, dim=0).unsqueeze(0)

        return stacked_values

    # def get_last_data_at_ref_time(self, reference_times, data_times, data_dict):

    #     indices = []
    #     idx = 0
    #     for ref_time in reference_times:
    #         while idx < len(data_times) - 1 and data_times[idx + 1] <= ref_time:
    #             idx += 1
    #         indices.append(idx)

    #     last_data_dict = {}
    #     for key in data_dict.keys():
    #         last_data_dict[key] = data_dict[key][indices]

    #     return last_data_dict

    # def get_ref_times(self):

        # ros_timestamp_ref = webcam1_buffer_copy[-1][0]  # last timestamp
        # t_sec_ref = ros_timestamp_ref.sec + ros_timestamp_ref.nanosec * 1e-9

        # delta_t_state = self.delta_timestamps['observation.state']  
        # delta_t_action = self.delta_timestamps['action'][:self.N_history]

        # ref_times = [t_sec_ref + dt for dt in delta_t_state]  # compute absolute timestamps

        # self.get_last_data_at_ref_time(ref_times, data_times, data_dict)

    def build_obs(self):
        
        # Make a snapshot copy inside lock
        with self.buffer_lock:
            buffer_copies = {k: list(v) for k, v in self.buffers.items()}

        # Convert data to tensors and create history
        data_tensors = {} # (N_history, ...)
        for k, buf in buffer_copies.items():
            is_image = "webcam" in k or "rgb" in k
            data_tensors[k] = self.stack_data(buf, is_image=is_image)

        # Rebuild state vector (N_hist, D) as in original dataset
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

        # Move data to device
        observation = {k: v.to(self.device, non_blocking=True) for k, v in observation.items()}

        # Apply custom transforms
        observation = self.tf_inference.transform(observation) # in/out: (B, N_h, ...)

        # Remove key "action"
        observation.pop("action", None)

        # PATCH to convert (B,N_hist, ...) in (B, ...) by taking last timestep
        for k, v in observation.items():
            if v.dim() >= 2:
                v = v[:, -1, ...].contiguous()
                observation[k] = v

        return observation

    def smooth_action(self, new_action, prev_action):
        
        # Split components

        pos_new  = new_action[:3]
        quat_new = new_action[3:7]
        grip_new = new_action[7]  # unchanged

        pos_prev  = prev_action[:3]
        quat_prev = prev_action[3:7]
        grip_prev = prev_action[7]

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

        # Wait at least 1 data for each buffer
        if (not(self.ready())):
            return

        # Build observation
        obs = self.build_obs()

        # Inference
        with torch.inference_mode():
            action = self.policy.select_action(obs) # (B, D) --> (B, D)
            # actions = self.policy.diffusion.generate_actions(obs) # (B, N_hist, D) --> (N_chunk, D)
            print("step: ", self.step)
            # print("policy action: ", action)
            self.step+=1

        # Move to CPU and convert to numpy
        action = action.squeeze(0).to("cpu")

        # Convert axis-angle to quaternion
        quat = CustomTransforms.axis_angle2quaternion(action[3:6])

        # Convert gripper in binary {0,1}
        action[-1] = CustomTransforms.gripper_continuous2discrete(action[-1])

        # Convert tensors to numpy
        action_np = action.numpy()
        quat_np = quat.numpy()

        # Build final action as expected by controller
        action_pre_tf = np.concatenate([action_np[:3], quat_np, [action_np[-1]]])
        print("action policy", action_pre_tf)
        
        # # Get previous action from buffer (take only the action values)
        # prev_action = np.array(self.buffers["action"][-1][1], dtype=np.float32)
        # print("prev_action", prev_action)

        # # Filter action
        # action_pre_tf = self.smooth_action(action_pre_tf, prev_action)
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



def main(args=None):

    rclpy.init(args=args)
    node = FrankaInference()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    # def build_obs(self):
        
    #     # Make a snapshot copy inside lock
    #     with self.buffer_lock:
    #         buffer_copies = {k: list(v) for k, v in self.buffers.items()}

    #     # Convert data to tensors and create history
    #     data_tensors = {} # (N_history, ...)
    #     for k, buf in buffer_copies.items():
    #         is_image = "webcam" in k or "rgb" in k
    #         data_tensors[k] = self.stack_data(buf, is_image=is_image)

    #     # Rebuild state vector (N_hist, D) as in original dataset
    #     state = torch.cat([
    #         data_tensors["q"],
    #         data_tensors["qdot"],
    #         data_tensors["tau"],
    #         data_tensors["cart_pos_curr"],
    #         data_tensors["cart_quat_curr"],
    #         data_tensors["gripper_state"],
    #         data_tensors["fext"],                 
    #     ], dim=-1)

    #     # Create the policy input dictionary as in original dataset
    #     observation = {
    #         "observation.state": state,
    #         "observation.images.front_cam1": data_tensors["webcam1"],
    #         "observation.images.front_cam2": data_tensors["webcam2"],
    #         "observation.images.front_cam3": data_tensors["webcam3"],
    #         "observation.images.gripper_camera": data_tensors["realsense_rgb"],
    #     }

    #     # Transforms need to include past actions in the state
    #     observation["action"] = data_tensors["action"]

    #     # Apply custom transforms
    #     observation = self.tf_inference.transform(observation) 

    #     # Remove key "action"
    #     observation.pop("action", None)

    #     # PATCH to convert (N_hist, ...) in (1, ...) by taking last timestep
    #     for k, v in observation.items():
    #         if v.dim() >= 2:
    #             v = v[-1, ...].contiguous()
    #             observation[k] = v.unsqueeze(0)

    #     # Move data to device
    #     observation = {k: v.to(self.device, non_blocking=True) for k, v in observation.items()}

    #     return observation