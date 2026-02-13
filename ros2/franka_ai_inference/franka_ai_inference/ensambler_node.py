#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from custom_msgs.msg import GripperWidth

from collections import deque
import torch
import numpy as np
from math import ceil



class FrankaEnsambler(Node):

    """
    ROS2 Node.
    """

    def __init__(self):

        super().__init__('FrankaEnsembler')

        # params
        self.alpha = 0.98

        # init
        self.action_buffer = [] # [ (t1, a1), (t2, a2) , ... ]
        self.active_trajectories = []

    def action_callback(self, msg):

        t_zero = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Creiamo la nuova traiettoria con i suoi timestamp
        new_trajectory = []
        for i, a_data in enumerate(msg.actions):
            t_action = round(t_zero + (i * self.dt), 4)
            new_trajectory.append((t_action, a_data))
        
        with self.lock:
            # Aggiungiamo la nuova traiettoria alla lista delle "traiettorie attive"
            self.active_trajectories.append(new_trajectory)
            
            # Pulizia: teniamo solo le ultime N traiettorie (es. le ultime 10)
            # per evitare di consumare troppa memoria
            if len(self.active_trajectories) > 10:
                self.active_trajectories.pop(0)

    def linear_interp(self):
        pass     

    def action_timer(self):

        now = round(self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9, 3)
        actions_to_average = []
        
        with self.lock:
            self.active_trajectories = [tr for tr in self.active_trajectories if tr[-1][0] > now] # Se una traiettoria è finita (l'ultimo timestamp è passato), rimuovila
            active_trajectories = list(self.active_trajectories) # copy to free callback fast

        # 1. Esplora tutte le traiettorie che abbiamo in memoria
        for traj in active_trajectories:
            # Cerca se questa traiettoria ha un'azione per "now"
            # (Usiamo una tolleranza di mezzo dt)
            for t_action, a_data in traj:
                if abs(t_action - now) < (self.dt / 2):
                    actions_to_average.append(a_data)
                    break

        # 3. Calcolo della Media (Temporal Ensembling)
        if actions_to_average:
            # Media semplice o pesata (es. dando più peso alle traiettorie più recenti)
            # Qui usiamo una media semplice per brevità
            final_action = np.mean(actions_to_average, axis=0)
            self.send_to_robot(final_action)




        # # Eventually smooth policy output
        # if self.smooth_output == True:

        #     # Get previous action from buffer (take only the action values)
        #     prev_action = np.array(self.buffer_past_actions[-1], dtype=np.float32)
        #     # print("prev_action", prev_action)

        #     # Filter action
        #     action_pre_tf = self.smooth_action(action_pre_tf, prev_action)
        #     # print("action policy filtered and applied", action_pre_tf)


        # quat = normalize_quat(quat)


        # # Save action in buffer safely
        # with self.buffer_lock:
        #     self.buffer_past_actions.append(action_pre_tf)

        # # Set cart pose action
        # cart_msg = PoseStamped()
        # cart_msg.header.stamp = self.get_clock().now().to_msg()
        # cart_msg.pose.position.x = float(action_pre_tf[0])
        # cart_msg.pose.position.y = float(action_pre_tf[1])
        # cart_msg.pose.position.z = float(action_pre_tf[2])
        # cart_msg.pose.orientation.x = float(action_pre_tf[3])
        # cart_msg.pose.orientation.y = float(action_pre_tf[4])
        # cart_msg.pose.orientation.z = float(action_pre_tf[5])
        # cart_msg.pose.orientation.w = float(action_pre_tf[6])

        # # Publish
        # self.cart_pose_action_pub.publish(cart_msg)

        # # Set gripper action
        # gripper_msg = GripperWidth()
        # gripper_msg.header.stamp = self.get_clock().now().to_msg()
        # gripper_msg.width = float(action_pre_tf[-1])

        # # Publish
        # self.gripper_action_pub.publish(gripper_msg)

def main(args=None):

    rclpy.init(args=args)
    node = FrankaEnsambler()

    try:
        node.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()