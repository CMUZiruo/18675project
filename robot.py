
import numpy as np
# Define the RobotArm class
class RobotArm:
    def __init__(self, link_length_1, 
                       link_length_2, 
                       link_width_2, 
                       q_init):
        self.L1 = link_length_1
        self.L2 = link_length_2
        self.W2 = link_width_2
        self.theta1 = q_init[0]
        self.theta2 = q_init[1]
        self.q_dot = np.zeros(2)
        self.contact_depth = 0.
        self.contact_height = 0.
        self.contact = False
        self.contact_vector = np.zeros(2)
        self.dynamic_object = {
            "x": 0.5,  # Initial x-coordinate
            "y": 0.7,  # Initial y-coordinate
        }
    
    def set_dynamic_object_position(self, x, y):
        # Set the initial position of the dynamic object
        self.dynamic_object["x"] = x
        self.dynamic_object["y"] = y

    def get_dynamic_object_position(self):
        # Get the current position of the dynamic object
        return self.dynamic_object["x"], self.dynamic_object["y"]

    def update_robot_states(self, control, sampling_rate):
        self.q_dot = control
        self.theta1 += self.q_dot[0] * sampling_rate
        self.theta2 += self.q_dot[1] * sampling_rate
        
    def update_contact_vector(self, contact_vector):
        self.contact_vector = contact_vector

    def get_contact_vector(self):
        return self.contact_vector

    def get_joint_angles(self):
        return np.array([self.theta1, self.theta2])

    def get_joint_velocities(self):
        return self.q_dot

    def get_contact_info(self):
        return (self.contact_depth, self.contact_height)

    def get_T_eff(self):
        x_end_effector, y_end_effector = self.forward_kinematics(self.theta1, self.theta2)
        theta = self.theta1 + self.theta2
        # Transformation matrix from end-effector frame to base frame
        T_end_effector_to_base = np.array([[np.cos(theta), -np.sin(theta), x_end_effector],
                                           [np.sin(theta), np.cos(theta), y_end_effector],
                                           [0, 0, 1]])
        return T_end_effector_to_base
    
    def get_T_contact_to_ee(self, obstacle_point):
        T_ee_to_base = self.get_T_eff()
        p_contact_in_eff = np.linalg.inv(T_ee_to_base) @ np.array([obstacle_point[0], obstacle_point[1], 1])
        if p_contact_in_eff[1] >= 0:
            T_contact_ee = np.array([[1, 0, p_contact_in_eff[0]], 
                                     [0, 1, p_contact_in_eff[1]], 
                                     [0, 0, 1]])
        else:
            T_contact_ee = np.array([[-1, 0, p_contact_in_eff[0]], 
                                     [0, -1, p_contact_in_eff[1]], 
                                     [0, 0, 1]])
        return T_contact_ee
    
    def get_obstacle_point_in_eff(self, obstacle_point):
        T_contact_to_ee =self.get_T_contact_to_ee(obstacle_point)
        return T_contact_to_ee[:2, 2]

    def compute_contact_info(self, obstacle_point):
        p_contact_in_eff = self.get_obstacle_point_in_eff(obstacle_point)
        if p_contact_in_eff[1] >= 0:
            contact_depth = self.W2/2 - p_contact_in_eff[1]
        else:
            contact_depth = self.W2/2 + p_contact_in_eff[1]
        contact_height = self.L2 + p_contact_in_eff[0]
        if contact_depth > 0:
            self.contact_depth = contact_depth
            self.contact_height = contact_height
        else: 
            self.contact_depth = 0.
            self.contact_height = 0.

    # Define forward kinematics 
    def forward_kinematics(self, theta1, theta2):
        """
        Compute the end-effector position using forward kinematics.
        
        Parameters:
        theta1 (float): Joint angle of the first joint in radians.
        theta2 (float): Joint angle of the second joint in radians.
        L1 (float): Length of the first link.
        L2 (float): Length of the second link.
        
        Returns:
        end_effector_pos (tuple): X and Y coordinates of the end-effector position.
        """
        x = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        y = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)
        return x, y

    # Jacobian calculation
    def contact_jacobian(self, theta1, theta2, contact_location):
        """
        Compute the Jacobian matrix for a 2-DOF planar robot arm.
        
        Parameters:
        theta1 (float): Joint angle of the first joint in radians.
        theta2 (float): Joint angle of the second joint in radians.
        L1 (float): Length of the first link.
        L2 (float): Length of the second link.
        
        Returns:
        J (numpy.ndarray): Jacobian matrix.
        """
        L1 = self.L1
        L2 = contact_location

        J11 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2)
        J12 = -L2 * np.sin(theta1 + theta2)
        J21 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        J22 = L2 * np.cos(theta1 + theta2)
        
        J = np.array([[J11, J12],
                    [J21, J22]])
        
        return J

    # Jacobian calculation
    def jacobian(self, theta1, theta2):
        """
        Compute the Jacobian matrix for a 2-DOF planar robot arm.
        
        Parameters:
        theta1 (float): Joint angle of the first joint in radians.
        theta2 (float): Joint angle of the second joint in radians.
        L1 (float): Length of the first link.
        L2 (float): Length of the second link.
        
        Returns:
        J (numpy.ndarray): Jacobian matrix.
        """
        L1 = self.L1
        L2 = self.L2

        J11 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2)
        J12 = -L2 * np.sin(theta1 + theta2)
        J21 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        J22 = L2 * np.cos(theta1 + theta2)
        
        J = np.array([[J11, J12],
                    [J21, J22]])
        
        return J


class Controller():
    def __init__(self, robot):
        # a dictionary that maps a target object to its corresponding sensing function/API
        self.sensing_fc_dict = {
                                "palm": self.get_predict_eff_pos,
                                "apple": self.get_predict_apple_pos,
                                "box": self.get_predict_box_pos,
                                "target_position": self.get_target_pos,
                                #"bottle": self.get_dynamic_object_position,
                                "bottle": self.get_predict_bottle_pos,
                                }     
        self.dynamic_object = {
            "x": 0.5,  # Initial x-coordinate
            "y": 0.7,  # Initial y-coordinate
        }  
        # get robot 
        self.robot = robot
        # object state
        self.current_xa = np.array([-1.0, 1.0]) # apple pos
        self.current_xb = np.array([0.3, 0.7]) # box pos

        self.current_xe = np.array([0., 0.])
        self.break_contact = False

        self.contact = False

        # MPC parameters
        self.control_dim = 2
        self.horizon = 5
        self.dt = 0.01
        # predict_state, not actual state
        self.state_ = np.array([0, 0]) 
        self.reward_functions = []
    
    def set_dynamic_object_position(self, x, y):
        # Set the initial position of the dynamic object
        self.dynamic_object["x"] = x
        self.dynamic_object["y"] = y

    def update_dynamic_object_position(self, dt):
        
        # Calculate the new position
        if self.dynamic_object["x"] >= 1.5:
            self.dynamic_object["x"] = -1.5
        self.dynamic_object["x"] +=  1 * dt
        self.dynamic_object["y"] +=  0 * dt
       

    def get_dynamic_object_position(self):
        # Get the current position of the dynamic object
        return self.dynamic_object["x"], self.dynamic_object["y"]

    def system_dynamics(self, state, control):
        return state + control * self.dt
    
    def update_predict_state(self, predict_state):
        self.state_ = predict_state
        
    def get_predict_state(self):
        return self.state_
    
    def optimization_objective(self, control_seq, state):
        total_reward = 0
        predicted_state = np.array(state)
        # print(control_seq)
        for i in range(self.horizon):

            # Get the current state of end-effector
            self.current_xe = self.get_predict_eff_pos()

            # Apply control action from the sequence
            control = control_seq[i * self.control_dim:(i + 1) * self.control_dim]
            
            # Update predicted state using system dynamics
            predicted_state = self.system_dynamics(predicted_state, control)
            # Update predicted state to the internal state, which requries for reward computation
            self.update_predict_state(predicted_state)
            # Calculate reward and accumulate
            # total_reward += reward_function("palm", "apple")*5
            total_reward += self.compute_reward(control)

        return -total_reward  # Negate to maximize the reward
    
    def update_reward_functions(self, reward_fc):
        self.reward_functions.append(reward_fc)

    def compute_reward(self, control):
        reward_sum = 0
        for reward_fc in self.reward_functions:
            fc_name = reward_fc["name"]
            fc_args = reward_fc["args"]
            fc_weight = reward_fc["weight"]
            reward = fc_name(fc_args[0], fc_args[1], self.sensing_fc_dict)
            reward_sum = reward_sum + fc_weight*reward
        reward_sum = reward_sum + 0.01*self.speed_reward(control)
        return reward_sum

    def speed_reward(self, control):
        if np.linalg.norm(control) > 1e-3:
            reward = -np.linalg.norm(control)
        else:
            reward = 0
        return reward

    """Sensing Function Call"""
    def get_predict_eff_pos(self):
        predcit_state = self.get_predict_state()
        x_e = self.robot.forward_kinematics(predcit_state[0], predcit_state[1])
        #print("xe",x_e)
        return np.array(x_e)

    def get_predict_apple_pos(self):
        # if the hand and apple is close at this time step
        # then at the next time step, apple follows the hand
        # otherwise, the apple remains unchanged
        distace = np.linalg.norm(self.current_xe - self.current_xa)
        if distace < 1e-3:
            xe = self.get_predict_eff_pos()
            self.current_xa = xe       
        return self.current_xa
    
    def get_predict_bottle_pos(self):
        # if the hand and apple is close at this time step
        # then at the next time step, apple follows the hand
        # otherwise, the apple remains unchanged
        self.current_bottle = [self.dynamic_object["x"], self.dynamic_object["y"]]
        distace = np.linalg.norm(self.current_xe - self.current_bottle)
        if distace < 1e-3 and not self.break_contact:
            xe = self.get_predict_eff_pos()

            self.dynamic_object = {
            "x": xe[0],  # Initial x-coordinate
            "y": xe[1],  # Initial y-coordinate

        }        
        return [self.dynamic_object["x"], self.dynamic_object["y"]]
            
    def get_predict_box_pos(self):
        # if the hand and apple is close at this time step
        # then at the next time step, apple follows the hand
        # otherwise, the apple remains unchanged
        distace = np.linalg.norm(self.current_xe - self.current_xb)
        if distace < 1e-3 and not self.break_contact:
            xe = self.get_predict_eff_pos()
            self.current_xb = xe       
        return self.current_xb
    
    def get_target_pos(self):
        return np.array([0.0, 1.0])

    def detect_contact(self):
        
        distance = np.linalg.norm(self.current_xe - self.current_bottle)
        if distance < 1e-3:
            self.contact = True
        # else:
        #     self.contact = False

        return self.contact

    def detect_safty(self, constrain):
        if constrain < 0.1:
            

def l2_distance_reward(name_obj_A, name_obj_B, sensing_fc_dict):
    """
    where name_obj_A and name_obj_B are selected from ['palm', 'apple', 'banana', 'box', 'bowl', 'drawer_handle', 'faucet_handle', 'drawer_center', 'rest_position'].
    This term sets a reward for minimizing l2_distance between name_obj_A and name_obj_B so they get closer to each other.
    rest_position is the default position for the palm when it's holding in the air.
    Default weight: 5
    """
    pos_A =  sensing_fc_dict[name_obj_A]()
    pos_B =  sensing_fc_dict[name_obj_B]()
    residual = pos_A - pos_B
    reward = np.linalg.norm(residual)

    return reward*(-1)

#define a function for safe_distance_constraint
def safe_distance_constraint(name_obj_A, name_obj_B, sensing_fc_dict):
    """
    where name_obj_A and name_obj_B are selected from ['palm', 'apple', 'banana', 'box', 'bowl', 'drawer_handle', 'faucet_handle', 'drawer_center', 'rest_position'].
    This term sets a constraint for maintaining a safe distance between name_obj_A and name_obj_B.
    rest_position is the default position for the palm when it's holding in the air.
    Default weight: 5
    """
    pos_A =  sensing_fc_dict[name_obj_A]()
    pos_B =  sensing_fc_dict[name_obj_B]()
    residual = pos_A - pos_B
    constraint = np.linalg.norm(residual) - 0.1

    return constraint

# Define a function as explained below
"""
def safe_speed_constraint(max_speed=1)
```
set the constraint on the robot speed. The robot velocity must not exceed the "max_speed"
default "max_speed"=0.1, but it can be modified according to distances between robots and external objects
"""
def safe_speed_constraint(max_speed=1):
    def constraint(control):
        if np.linalg.norm(control) > max_speed:
            return 1
        else:
            return 0
    return constraint          




