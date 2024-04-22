import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json
import openai
import time
from collections import deque

# numpy version: 1.26.3
# json version: 2.0.9
# openai version: 0.28.0
# matplotlib version: 3.8.2
# scipy version: 1.12.0


from scipy.optimize import minimize
import numpy as np
import time
import matplotlib.pyplot as plt
import json
import openai
from robot import RobotArm, Controller
from robot import l2_distance_reward
from animation import RobotAnimation
from llm_utils import load_instruction_file, llm_call


# Constants
L1 = 1.0  # Length of the first link
L2 = 0.8  # Length of the second link
W2 = 0.2
# Initial joint angles
theta1_initial = np.deg2rad(90)
theta2_initial = np.deg2rad(180)
#theta1_initial = np.deg2rad(30)
#theta2_initial = np.deg2rad(30)

# Create an instance of RobotArm
robot_arm = RobotArm(link_length_1=L1, 
                     link_length_2=L2,
                     link_width_2 = W2,
                     q_init=np.array([theta1_initial, theta2_initial]))
# Create an instance of Controller
controller = Controller(robot_arm)
# Create an instance of RobotAnimation
robot_ani = RobotAnimation(robot_arm, controller)



def reset_reward():
    """
    This function resets the reward to default values.
    """
    controller.reward_functions = []
    # controller.current_xe = np.array([0., 0.])

def set_l2_distance_reward(name_obj_A, name_obj_B):
    """
    where name_obj_A and name_obj_B are selected from ['palm', 'apple', 'banana', 'box', 'bowl', 'drawer_handle', 'faucet_handle', 'drawer_center', 'rest_position'].
    This term sets a reward for minimizing l2_distance between name_obj_A and name_obj_B so they get closer to each other.
    rest_position is the default position for the palm when it's holding in the air.
    Default weight: 5
    """
    func_params = {"name": l2_distance_reward,
                   "args": [name_obj_A, name_obj_B],
                   "weight": 5}
    controller.update_reward_functions(func_params)

# define a function for safe_distance_constraint
# def safe_distance_constraint(name_obj_A, name_obj_B, sensing_fc_dict):
#     """
#     where name_obj_A and name_obj_B are selected from ['palm', 'apple', 'banana', 'box', 'bowl', 'drawer_handle', 'faucet_handle', 'drawer_center', 'rest_position'].
#     This term sets a constraint for maintaining a safe distance between name_obj_A and name_obj_B.
#     rest_position is the default position for the palm when it's holding in the air.
#     Default weight: 5
#     """
#     pos_A =  sensing_fc_dict[name_obj_A]()
#     print(pos_A)
#     pos_B =  sensing_fc_dict[name_obj_B]()
#     print(pos_B)
#     residual = pos_A - pos_B
#     constraint = np.linalg.norm(residual) - 0.1

#     return constraint


def safe_distance_constraint(name_obj_A, name_obj_B, sensing_fc_dict):
    """
    where name_obj_A and name_obj_B are selected from ['palm', 'apple', 'banana', 'box', 'bowl', 'drawer_handle', 'faucet_handle', 'drawer_center', 'rest_position'].
    This term sets a constraint for maintaining a safe distance between name_obj_A and name_obj_B.
    rest_position is the default position for the palm when it's holding in the air.
    Default weight: 5
    """
    constraint = sensing_fc_dict.values()
    print(constraint)
    controller.safe_distance = list(constraint)[0]
    robot_arm.avoid_object = name_obj_B

def safe_speed_constraint(max_speed=1):
    def constraint(control):
        if np.linalg.norm(control) > max_speed:
            return 1
        else:
            return 0
    return constraint   

def execute_plan(duration=4):
    """
    This function sends the parameters to the robot and execute the plan for `duration` seconds, default to be 2
    """
    duration = 3
    # control_seq = np.random.rand(controller.control_dim*controller.horizon)
    control_seq = np.zeros(controller.control_dim*controller.horizon)
    control_bounds = [(-1.0, 1.0)] * (controller.control_dim*controller.horizon)
    rewards = []
    initial_reward = controller.compute_reward(control_seq[0:2])
    print("Initial reward: {}".format(initial_reward))
    # Calculate the number of time steps
    num_steps = int(duration / controller.dt)
    actual = np.zeros(num_steps)
    
    for i in range(num_steps):
        
        
        # detect if contact
        controller.detect_contact()
        
        # if not contact, update dynamic object position
        if controller.contact == False:
            controller.update_dynamic_object_position(controller.dt)

        state = robot_arm.get_joint_angles()
        #set a position for the dynamic object that was defined in robot.py
        # Optimize the control sequence

        result = minimize(controller.optimization_objective, 
                          control_seq, 
                          state,
                          bounds=control_bounds)
        # Get the optimal control action for the current time step
        control_seq = result.x
        optimal_control = control_seq[0:controller.control_dim]
        # print(np.rad2deg(optimal_control))

        out = np.linalg.norm(optimal_control)

        

        #determine the position of avoiding object
        if robot_arm.avoid_object == "bottle":
            pos_B = controller.current_bottle
        elif robot_arm.avoid_object == "box":
            pos_B = controller.current_xb
        elif robot_arm.avoid_object == "apple":
            pos_B = controller.current_xa
        else:
            pos_B = []

        # update robot state
        robot_arm.update_robot_states(optimal_control, controller.dt, controller.current_xe, pos_B, controller.safe_distance)

        #robot_arm.detect_safty(controller.current_xe, controller.current_xb, controller.safe_distance)

        # Get the optimized reward value (negative of the total_reward)
        optimized_reward = controller.compute_reward(optimal_control)
        

        pid_reward = controller.update_pid(optimized_reward,actual[i-1])
        #print("optimized_reward",optimized_reward)
        actual[i] = actual[i-1]+pid_reward*controller.dt
        rewards.append(pid_reward)
        print("optimized_reward",pid_reward)
        if np.abs(pid_reward) < 1e-2:
            break
        #print(actual)
        #print("[{}/20]Reward: {}".format(i+1, -distace_reward(xe, get_apple_pos())))
        plt.pause(controller.dt)
    print("Final Reward: {}".format(rewards[-1]))
    return rewards 


"""
GPT-based LLM model Section
"""
#Enter your OpenAI key here
openai.api_key = "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"
print("Successfully OpenAI key authorization")

descriptor = "descriptor_order.txt"
coder = "coder_order.txt"
motion_descriptor = load_instruction_file(descriptor)
print("\nWe are using motion descriptor file:\n{}".format(descriptor))
reward_coder = load_instruction_file(coder)
print("\nWe are using coder file:\n{}".format(coder))
md_temp = 0.5 # motion descriptor temperature
rc_temp = 0.2 # reward coder temperature


# # Open the JSON file in read mode
# with open("functions.json", "r") as json_file:
#     # Parse the JSON data
#     fc_json = json.load(json_file)
# # Now 'data' contains the parsed JSON data as a Python dictionary or list
# fc_rewards = fc_json["rewards"]
# fc_constraints = fc_json["constraints"]

# md: motion descriptor
# rc: reward coder

#curr_pos = RobotArm.get_T_eff()
#curr_pos = robot_arm.get_T_eff()

md_messages = [
    {"role": "system", "content": motion_descriptor}
]
rc_messages = [
    {"role": "system", "content": reward_coder},
]
# first call, to instruct the system what to do
# providing the model with motion or coder generation template
md_response = llm_call(md_messages, temperature=md_temp)
rc_response = llm_call(rc_messages, temperature=rc_temp)

md_response_mess = md_response["choices"][0]["message"]
# print("Message response from Motion Descritor:\n{}".format(md_response_mess))
rc_response_mess = rc_response["choices"][0]["message"]
# print("Message response from Reward Coder:\n{}".format(rc_response_mess))
# add the newly-created reponse to the previous meassages or conversation
md_messages.append(md_response_mess)
rc_messages.append(rc_response_mess)
for i in range(100):
    # Prompt the user for input and store it in a variable
    user_input = input("User: ")
    # add new user command to the Motion Descriptor meassages first
    md_messages.append({"role": "system", "content": user_input})

    print("\nWaiting for the next LLM call due to free trial usage limits")
    # print("\nWe are using ",openai.Model)
    time.sleep(5)
    # Motion descriptor call
    md_response = llm_call(md_messages, temperature=md_temp)
    md_response_mess = md_response["choices"][0]["message"]
    md_response_content = md_response_mess["content"]
    print("\nMotion Descriptor:\n{}".format(md_response_content))
    # From the motion descriptor response message, querying it to the reward corder LLM model call
    rc_messages.append({"role": "system", 
                        "content": md_response_content})

    print("\nWaiting for the next LLM call due to free trial usage limits")
    time.sleep(5)
    # Reward coder call
    rc_response = llm_call(rc_messages, temperature=rc_temp)
    print(rc_response)
    rc_response_mess = rc_response["choices"][0]["message"]
    rc_response_content = rc_response_mess["content"]
    print("\nCoder:\n{}".format(rc_response_content))
    print("\n\n")

    # add the newly-created reponse to the previous meassages or conversation
    md_messages.append(md_response_mess)
    rc_messages.append(rc_response_mess)

    rc_response_content = rc_response_content.replace("```python", "").strip('` \n')
    code = "{}".format(rc_response_content)
    # print(code)
    # Create a local namespace for execution
    curr_pos = robot_arm.get_T_eff()
    local_namespace = {}
    global_namespace = {"reset_reward": reset_reward, 
                        "set_l2_distance_reward": set_l2_distance_reward, 
                        "safe_distance_constraint": safe_distance_constraint,
                        "safe_speed_constraint": safe_speed_constraint,
                        "execute_plan": execute_plan,
                        "sensing_fc_dict": {"palm": robot_arm.get_T_eff, 
                                            "apple": controller.get_predict_apple_pos,
                                            "box": controller.get_predict_box_pos,
                                            "target": controller.get_target_pos,
                                            #"bottle": controller.get_dynamic_object_position,
                                            "bottle": controller.get_predict_bottle_pos
                                            },
                        "curr_pos": curr_pos}
    # Execute the code
    exec(code, global_namespace, local_namespace)
    controller.break_contact = True
    # Retrieve the 'rewards' variable from the local namespace
    rewards = local_namespace.get("rewards", None)
     
_, ax = plt.subplots()
ax.plot(rewards, linewidth=2, color='red')
ax.set_xlim([0, np.max(len(rewards))])
ax.set_xlabel('Time step')
ax.set_ylabel('Reward')
ax.set_title('Reward cruve')
plt.show()