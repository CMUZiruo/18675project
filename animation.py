import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import time
from collections import deque

class RobotAnimation():
    def __init__(self, robot, controller) -> None:
        self.robot = robot
        self.controller = controller
        self.apple_target = self.controller.get_predict_apple_pos()
        self.box_target = self.controller.get_predict_box_pos()
        #self.bottle_target = self.controller.get_predict_bottle_pos()
        self.bottle_target = self.controller.get_predict_bottle_pos()
        self.dynamic_object_target = self.controller.get_dynamic_object_position()

        # Create the figure and axis
        fig, ax_animation = plt.subplots(1, 1, figsize=(5, 5))

        ### Define objects for animation axis ###
        ax_animation.set_xlim(-1.5, 1.5)
        ax_animation.set_ylim(-1.5, 2)
        ax_animation.set_xlabel('x [m]', fontsize=14)
        ax_animation.set_ylabel('y [m]', fontsize=14)
        # Set the font size of the tick labels on the x and y axes
        ax_animation.tick_params(axis='x', labelsize=14)
        ax_animation.tick_params(axis='y', labelsize=14)
        ax_animation.set_title('2-DoF robot arm', fontsize=14)
        ax_animation.set_aspect('equal')
        # Initialize the robot arm links
        self.link1, = ax_animation.plot([], [], 'b-', lw=6, marker='o', markersize=10)
        self.link2, = ax_animation.plot([], [], 'b-', lw=4, marker='o', markersize=8)

        # Add a mark of the goal / target location
        self.apple = ax_animation.scatter(self.apple_target[0], 
                                          self.apple_target[1],
                                          color='red', 
                                          marker='o', 
                                          s=200)
        
        self.bottle = ax_animation.scatter(self.dynamic_object_target[0], 
                                          self.dynamic_object_target[1],
                                          color='green', 
                                          marker='o', 
                                          s=200)
                                          
        # self.bottle = ax_animation.scatter(self.bottle_target[0],
        #                                    self.bottle_target[1],
        #                                    color = 'blue',
        #                                    marker = 'o',
        #                                    s = 150)
        
        self.box = ax_animation.scatter(self.box_target[0], 
                                        self.box_target[1], 
                                        color='green', 
                                        marker='s', 
                                        s=250)
        
        self.target_location = ax_animation.scatter(
                                        0.0, 
                                        1.5, 
                                        color='black', 
                                        marker='x', 
                                        s=250)

        self.ani = FuncAnimation(fig, 
                                 self.update, 
                                 frames=None, 
                                 interval=30, 
                                 cache_frame_data=False)
        
        plt.show(block=False)

    def update(self, frame):
        # add robot parameters
        L1 = self.robot.L1
        q = self.robot.get_joint_angles()
        theta1 = q[0]
        xe = self.robot.get_T_eff()[:2, 2]
        xa = self.controller.get_predict_apple_pos()
        xb = self.controller.get_predict_box_pos()
        xc = self.controller.get_predict_bottle_pos()

        # Update link positions for animation
        self.link1.set_data([0, L1  * np.cos(theta1)], [0, L1  * np.sin(theta1)])
        self.link2.set_data([L1  * np.cos(theta1), xe[0]], [L1  * np.sin(theta1), xe[1]])
        self.apple.set_offsets(np.array([[xa[0], xa[1]]]))
        self.box.set_offsets(np.array([[xb[0], xb[1]]]))
        self.bottle.set_offsets(np.array([[xc[0], xc[1]]]))

if __name__ == "__main__":
    from robot import RobotArm

    # Constants
    L1 = 1.0  # Length of the first link
    L2 = 0.8  # Length of the second link
    W2 = 0.2
    # Initial joint angles
    # theta1_initial = np.deg2rad(90)
    # theta2_initial = np.deg2rad(-60)
    theta1_initial = np.deg2rad(30)
    theta2_initial = np.deg2rad(30)

    # Create an instance of RobotArm
    robot = RobotArm(link_length_1=L1, 
                     link_length_2=L2,
                     link_width_2 = W2,
                     q_init=np.array([theta1_initial, theta2_initial]))
    
    robot_ani = RobotAnimation(robot)
    
    # Calculate the number of time steps
    duration = 2
    dt = 0.01
    num_steps = int(10 / dt)
    for i in range(num_steps):
        control = np.array([0.1, 0.1])
        robot.update_robot_states(control, dt)
        plt.pause(dt)

    print("Finished")