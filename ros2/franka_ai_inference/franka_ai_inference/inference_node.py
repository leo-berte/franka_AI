#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

from collections import deque
import cv2
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

from franka_ai.dataset.transforms import CustomTransforms


# TODO:
# 1) path checkpoint from argparse o launcher


class FrankaInference(Node):

    def __init__(self):

        super().__init__('FrankaInference')

        # Declare ROS2 parameters with default values
        self.declare_parameter('policy_rate', 30)

        # Get parameter values
        self.policy_rate = self.get_parameter('policy_rate').get_parameter_value().integer_value

        # Parameters
        device = "cuda"
        N_history = 3
        N_chunk = 3
        pretrained_policy_path = 

        # Setup cv2 bridge
        self.bridge = CvBridge()

        # deque
        self.webcam1_buffer = deque(maxlen=N_history)
        self.webcam2_buffer = deque(maxlen=N_history)
        self.webcam3_buffer = deque(maxlen=N_history)
        self.realsense_rgb_buffer = deque(maxlen=N_history)
        self.realsense_depth_buffer = deque(maxlen=N_history)
        self.joint_state_buffer = deque(maxlen=N_history)
        self.gripper_state_buffer = deque(maxlen=N_history)
        self.fext_buffer = deque(maxlen=N_history)
        self.cart_pos_curr_buffer = deque(maxlen=N_history)
        self.cart_pos_filtered_command_buffer = deque(maxlen=N_history)
        self.gripper_command_buffer = deque(maxlen=N_history)

        # Subscribers
        self.webcam1_sub = self.create_subscription(CompressedImage, '/webcam1/image_raw/compressed', self.webcam1_callback, 10)
        self.webcam2_sub = self.create_subscription(CompressedImage, '/webcam2/image_raw/compressed', self.webcam2_callback, 10)
        self.webcam3_sub = self.create_subscription(CompressedImage, '/webcam3/image_raw/compressed', self.webcam3_callback, 10)
        self.realsense_rgb_sub = self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.realsense_rgb_callback, 10)
        self.realsense_depth_sub = self.create_subscription(CompressedImage, '/camera/camera/depth/image_rect_raw', self.realsense_depth_callback, 10)
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

    def webcam1_callback(self, msg):

        frame = bridge.imgmsg_to_cv2(msg)
        self.webcam1_buffer.append((msg.header.stamp, frame))

    def webcam2_callback(self, msg):

        pass

    def webcam3_callback(self, msg):

        pass
    
    def state_callback(self, msg):

        state_vec = parse_state(msg)
        self.state_buffer.append((msg.header.stamp, state_vec))

    def build_obs(self):
        
        # Convert to float32 with image from channel first in [0,255] to channel last in [0,1]
        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.images.front_cam1": webcam1,
            "observation.images.front_cam2": webcam2,
            "observation.images.front_cam3": webcam3,
            "observation.images.gripper_camera": realsense_rgb,
        }

        return observation
    
    def inference_timer(self):
            
            # build obs
            obs = self.build_obs()

            # inference
            with torch.inference_mode():
                action = self.policy.select_action(obs)

            # Prepare the action for the environment
            numpy_action = action.squeeze(0).to("cpu").numpy()

            # publish cart pose action
            msg = 
            # msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.cart_pos_action_pub.publish(msg)

            # publish gripper action
            msg = 
            # msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.gripper_action_pub.publish(msg)

def main(args=None):

    rclpy.init(args=args)
    node = FrankaInference()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()