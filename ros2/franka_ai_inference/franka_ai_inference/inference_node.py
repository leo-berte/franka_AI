#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from franka_msgs.msg import GripperWidth

from cv_bridge import CvBridge
from collections import deque
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.utils import get_configs_dataset


# TODO:
# 1) path checkpoint from argparse o launcher
# 2) se uso 2 repo, devo dare percorso assoluto per: get_configs_dataset("configs/dataset.yaml")
# 3) check append di past actions
# import gripper custom
# uso filtered o output diretto della policy?
# build timestamp
# capire se output è traiettoria di azioni e come pubblicarla
# traj stitching



class FrankaInference(Node):

    def __init__(self):

        super().__init__('FrankaInference')

        # Declare ROS2 parameters with default values
        self.declare_parameter('policy_rate', 30.0)

        # Get parameter values
        self.policy_rate = self.get_parameter('policy_rate').value
        pretrained_policy_path = "/workspace/outputs/checkpoints/today_outliers_DP_2025-11-26_11-54-34/best_model.pt"

        # Get configs about dataloader and dataset 
        dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset("configs/dataset.yaml")

        # Set yaml parameters
        self.device = dataloader_cfg["device"]
        self.N_history = dataloader_cfg["N_history"]
        self.N_chunk = dataloader_cfg["N_chunk"]

        # Get dataset state composition via indeces 
        state_ranges = dataset_cfg["state_slices"]
        self.state_slices = {k: slice(v[0], v[1]) for k, v in state_ranges.items()}

        # Prepare transforms for inference
        self.tf_inference = CustomTransforms(
            dataloader_cfg=dataloader_cfg,
            dataset_cfg=dataset_cfg,
            transforms_cfg=transforms_cfg,
            train=False
        )

        # Setup cv2 bridge
        self.bridge = CvBridge()

        # deque
        self.webcam1_buffer = deque(maxlen=self.N_history)
        self.webcam2_buffer = deque(maxlen=self.N_history)
        self.webcam3_buffer = deque(maxlen=self.N_history)
        self.realsense_rgb_buffer = deque(maxlen=self.N_history)
        self.realsense_depth_buffer = deque(maxlen=self.N_history)
        self.q_buffer = deque(maxlen=self.N_history)
        self.qdot_buffer = deque(maxlen=self.N_history)
        self.tau_buffer = deque(maxlen=self.N_history)
        self.gripper_state_buffer = deque(maxlen=self.N_history)
        self.fext_buffer = deque(maxlen=self.N_history)
        self.cart_pos_curr_buffer = deque(maxlen=self.N_history) 
        self.cart_quat_curr_buffer = deque(maxlen=self.N_history)
        # self.cart_pos_filtered_command_buffer = deque(maxlen=self.N_history)
        # self.cart_quat_filtered_command_buffer = deque(maxlen=self.N_history)
        # self.gripper_command_buffer = deque(maxlen=self.N_history)
        self.action_buffer = deque(maxlen=self.N_history)

        # Subscribers
        self.webcam1_sub = self.create_subscription(CompressedImage, '/webcam1/image_raw/compressed', self.webcam1_callback, 10)
        self.webcam2_sub = self.create_subscription(CompressedImage, '/webcam2/image_raw/compressed', self.webcam2_callback, 10)
        self.webcam3_sub = self.create_subscription(CompressedImage, '/webcam3/image_raw/compressed', self.webcam3_callback, 10)
        self.realsense_rgb_sub = self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.realsense_rgb_callback, 10)
        self.realsense_depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.realsense_depth_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/cartesian_impedance/joint_state', self.joint_state_callback, 10)
        self.gripper_state_sub = self.create_subscription(GripperWidth, '/panda_gripper/width', self.gripper_state_callback, 10)
        self.fext_sub = self.create_subscription(WrenchStamped, '/cartesian_impedance/f_ext_cart', self.fext_callback, 10)
        self.cart_pos_curr_sub = self.create_subscription(PoseStamped, '/cartesian_impedance/cartesian_pos_curr', self.cart_pos_curr_callback, 10)
        # self.cart_pos_filtered_action_sub = self.create_subscription(PoseStamped, '/cartesian_impedance/cartesian_pos_des_filt', self.cart_pos_filtered_action_callback, 10)
        # self.gripper_action_sub = self.create_subscription(GripperWidth, '/panda_gripper/gripper_command', self.gripper_action_callback, 10)
        
        # Publisher
        self.cart_pos_action_pub = self.create_publisher(PoseStamped, '/cartesian_impedance/equilibrium_pose', 10)
        self.gripper_action_pub = self.create_publisher(GripperWidth, '/panda_gripper/gripper_command', 10)

        # Timer according to framerate
        self.timer = self.create_timer(1.0 / self.policy_rate, self.inference_timer)

        # Load policy
        self.policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
        self.policy.reset() # reset the policy to prepare for rollout

        # Extract input features keys
        self.policy_input_features_keys = self.policy.config.input_features.keys()
        self.dataset_input_features_keys = dataset_cfg["features"]["VISUAL"] + dataset_cfg["features"]["STATE"]

        self.get_logger().info("FrankaInference node initialized successfully")

    def webcam1_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        rgb = rgb.astype('float32') / 255.0  # convert to float32 and normalize [0, 1]
        self.webcam1_buffer.append((msg.header.stamp, rgb))

    def webcam2_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        rgb = rgb.astype('float32') / 255.0  # convert to float32 and normalize [0, 1]
        self.webcam1_buffer.append((msg.header.stamp, rgb))

    def webcam3_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        rgb = rgb.astype('float32') / 255.0  # convert to float32 and normalize [0, 1]
        self.webcam1_buffer.append((msg.header.stamp, rgb))

    def realsense_rgb_callback(self, msg):

        rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8') # convert to RGB to match dataset format
        rgb = rgb.astype('float32') / 255.0  # convert to float32 and normalize [0, 1]
        self.webcam1_buffer.append((msg.header.stamp, rgb))
    
    def joint_state_callback(self, msg):

        # Joint positions, velocities, efforts
        self.q_buffer.append((msg.header.stamp, list(msg.position)))
        self.qdot_buffer.append((msg.header.stamp, list(msg.velocity)))
        self.tau_buffer.append((msg.header.stamp, list(msg.effort)))

    def gripper_state_callback(self, msg):
        
        # Gripper width
        self.gripper_state_buffer.append((msg.header.stamp, msg.width))

    def fext_callback(self, msg):

        # Extract gripper forces
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        tx = msg.wrench.torque.x
        ty = msg.wrench.torque.y
        tz = msg.wrench.torque.z

        self.fext_buffer.append((msg.header.stamp, [fx, fy, fz, tx, ty, tz]))

    def cart_pos_curr_callback(self, msg):

        # Cart pose
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        self.cart_pos_curr_buffer.append((msg.header.stamp, [px, py, pz]))
        self.cart_ori_curr_buffer.append((msg.header.stamp, [qx, qy, qz, qw]))

    def ready(self):
        return (
            len(self.webcam1_buffer) > 0 and
            len(self.webcam2_buffer) > 0 and
            len(self.webcam3_buffer) > 0 and
            len(self.realsense_rgb_buffer) > 0 and
            len(self.realsense_depth_buffer) > 0 and
            len(self.q_buffer) > 0 and
            len(self.qdot_buffer) > 0 and
            len(self.tau_buffer) > 0 and
            len(self.gripper_state_buffer) > 0 and
            len(self.fext_buffer) > 0 and
            len(self.cart_pos_curr_buffer) > 0 and
            len(self.cart_quat_curr_buffer) > 0
        )

    def stack_data(self, buffer, is_image=False):

        # [(t0, x0), (t1, x1), (t2, x2)] 
        # [x0, x1, x2, x2, x2] 

        # Get last N_history data
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
            values = [torch.from_numpy(v).float() for v in values]
            values = [v.permute(2, 0, 1) for v in values] # permute images from HWC to CHW
        else:
            values = [torch.tensor(v, dtype=torch.float32) for v in values]

        # Stack to (N_history, ... )
        values = torch.stack(values, dim=0)

        return values

    def validate_obs_keys(self, obs):

        # Get keys from policy and from observation vector
        obs_keys = set(obs.keys())
        dataset_input_keys = set(self.dataset_input_features_keys)
        policy_input_keys = set(self.policy_input_features_keys)

        # Check: all necessary keys are present
        missing = policy_input_keys - obs_keys
        if missing:
            raise ValueError(f"Missing observations required by policy: {missing}")

        # Check: no extra keys
        extra = obs_keys - policy_input_keys
        if extra:
            raise ValueError(f"Extra observations are present as input to the policy: {extra}")
        
        # Check policy input keys match dataset.yaml
        if dataset_input_keys != policy_input_keys:
            raise ValueError(
                f"Dataset features and policy input_features differ!\n"
                f"dataset={dataset_input_keys}\n"
                f"policy={policy_input_keys}"
            )

    def build_obs(self):
        
        # Convert data to tensors and create history
        webcam1 = self.stack_data(self.webcam1_buffer, is_image=True)
        webcam2 = self.stack_data(self.webcam2_buffer, is_image=True)
        webcam3 = self.stack_data(self.webcam3_buffer, is_image=True)
        realsense_rgb = self.stack_data(self.realsense_rgb_buffer, is_image=True)
        q = self.stack_data(self.q_buffer)
        qdot = self.stack_data(self.qdot_buffer)
        tau = self.stack_data(self.tau_buffer)
        gripper_state = self.stack_data(self.gripper_state_buffer)
        fext = self.stack_data(self.fext_buffer)
        cart_pos_curr = self.stack_data(self.cart_pos_curr_buffer)
        cart_quat_curr = self.stack_data(self.cart_quat_curr_buffer)
        action = self.stack_data(self.action_buffer)

        # Rebuild state vector (N_hist, D)
        state = torch.cat([
            q,
            qdot,
            tau,
            cart_pos_curr,
            cart_quat_curr,
            gripper_state,
            fext                 
        ], dim=-1)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.images.front_cam1": webcam1,
            "observation.images.front_cam2": webcam2,
            "observation.images.front_cam3": webcam3,
            "observation.images.gripper_camera": realsense_rgb,
        }

        # Check features match policy and dataset.yaml
        self.validate_obs_keys(observation)

        # Transforms will include past actions in the state
        observation["action"] = action

        # Apply custom transforms
        observation = self.tf_inference.transform(observation) 

        # Remove key "action"
        observation.pop("action", None)

        # Add extra (empty) batch dimension, required to forward the policy
        observation = {k: v.unsqueeze(0) for k, v in observation.items()}

        # Move data to device
        observation = {k: v.to(self.device, non_blocking=True) for k, v in observation.items()}

        return observation
    
    def inference_timer(self):

        # Wait at least 1 data for each buffer
        if (not(self.ready())):
            return

        # Build observation
        obs = self.build_obs()

        # Inference
        with torch.inference_mode():
            action = self.policy.select_action(obs) # che dim ha????

        # Convert to numpy
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Convert axis-angle to quaternion
        quat = CustomTransforms.axis_angle2quaternion2(numpy_action[3:7])

        # Save action in buffer
        action = numpy_action[:3] + quat + numpy_action[-1]
        self.action_buffer.append((self.get_clock().now().to_msg(), action))
        # self.action_buffer.append((self.get_clock().now().nanoseconds, action)) 

        # Set cart pose action
        cart_msg = PoseStamped()
        cart_msg.header.stamp = self.get_clock().now().to_msg()
        cart_msg.pose.position.x = float(numpy_action[0])
        cart_msg.pose.position.y = float(numpy_action[1])
        cart_msg.pose.position.z = float(numpy_action[2])
        cart_msg.pose.orientation.x = float(quat[0])
        cart_msg.pose.orientation.y = float(quat[1])
        cart_msg.pose.orientation.z = float(quat[2])
        cart_msg.pose.orientation.w = float(quat[3])

        # Publish
        self.cart_pos_action_pub.publish(cart_msg)

        # Set gripper action
        gripper_msg = GripperWidth()
        gripper_msg.header.stamp = self.get_clock().now().to_msg()
        gripper_msg.width = float(numpy_action[-1])

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