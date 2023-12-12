import numpy as np
from assignment_7_helper import World

np.set_printoptions(precision=3, suppress=True)


def stack():
    """
    function to stack objects
    :return: average height of objects
    """
    # DO NOT CHANGE: initialize world
    env = World()

    # ============================================
    # YOUR CODE HERE:
    
    def push(push_lift, push_from, push_to):
        env.robot_command([push_lift])
        env.robot_command([push_from])
        env.robot_command([push_to])
        
    def grasp(grasp_lift, grasp_pos, grasp_close):
        env.robot_command([grasp_lift])
        env.robot_command([grasp_pos])
        env.robot_command([grasp_close])
    
    def flip(move_up, move_to, drop):
        env.robot_command([move_up])
        env.robot_command([move_to])
        env.robot_command([drop])
    
    def stack(stack_lift, stack_grasp, stack_grasp_close, stack_grasp_lift, stack_grasp_move, stack_pos):
        env.robot_command([stack_lift])
        env.robot_command([stack_grasp])
        env.robot_command([stack_grasp_close])
        env.robot_command([stack_grasp_lift])
        env.robot_command([stack_grasp_move])
        env.robot_command([stack_pos])
        
    def stack_box(stack_grasp_lift, stack_grasp_move, stack_pos):
        env.robot_command([stack_grasp_lift])
        env.robot_command([stack_grasp_move])
        env.robot_command([stack_pos])
    
    obj_state = env.get_obj_state()   
    push_lift = np.zeros((5))
    stack_lift = np.zeros((5))

    for i in range(obj_state.shape[0]):
        # last box
        if i == obj_state.shape[0] - 1:
            grasp_lift[:3] = obj_state[i][:3]
            grasp_pos = grasp_lift + np.array([0., 0., -0.025, 0., 0.])
            grasp_close = grasp_pos + np.array([0., 0., 0., 0., -0.2])
            grasp(grasp_lift, grasp_pos, grasp_close)
            stack_grasp_lift = grasp_close + np.array([0., 0., 0.1*(i+1), 0., 0.])
            stack_grasp_move = np.array([-0.15, 0., 0.08*(i+1), 0., 0.])
            stack_pos = stack_grasp_move + np.array([0., 0., 0., 0., 0.2])
            stack_box(stack_grasp_lift, stack_grasp_move, stack_pos)
            env.home_arm()
            break
        
        # push
        push_lift[:3] = obj_state[i][:3] + np.array([0.02, 0.08, 0.025])
        push_from = push_lift + np.array([0., 0., -0.05, 0., 0.])
        push_to = push_from + np.array([0., -0.18, 0., 0., 0.])
        push(push_lift, push_from, push_to)
        # grasp to flip
        grasp_lift = push_to + np.array([0., 0., 0.08, 0., 0.2])
        grasp_pos = grasp_lift + np.array([-0.02, -0.065, -0.08, 0., 0.])
        grasp_close = grasp_pos + np.array([0., 0., 0., 0., -0.2])
        grasp(grasp_lift, grasp_pos, grasp_close)
        # flip
        move_up = grasp_close + np.array([0., 0., 0.13, 0., 0.])
        move_to = move_up + np.array([-obj_state[i][0], -0.18, 0., 0., 0.])
        drop = move_to + np.array([0., 0., 0., 0., 0.2])
        flip(move_up, move_to, drop)
        # stack
        obj_state = env.get_obj_state()
        print(obj_state)
        stack_lift[:3] = obj_state[i][:3] + np.array([0., 0., 0.05])
        stack_lift[-1] = 0.2
        stack_grasp = stack_lift + np.array([0., 0., -0.05, 0., 0.])
        stack_grasp_close = stack_grasp + np.array([0., 0., 0., 0., -0.2])
        stack_grasp_lift = stack_grasp_close + np.array([0., 0., 0.1*(i+1), 0., 0.])
        stack_grasp_move = np.array([-0.15, 0., 0.08*(i+1), 0., 0.])
        stack_pos = stack_grasp_move + np.array([0., 0., 0., 0., 0.2])
        stack(stack_lift, stack_grasp, stack_grasp_close, stack_grasp_lift, stack_grasp_move, stack_pos)
        
        env.home_arm()   

    # ============================================
    # DO NOT CHANGE: getting average height of objects:
    obj_state = env.get_obj_state()
    avg_height = np.mean(obj_state[:, 2])
    print("Average Object Height: {:4.3f}".format(avg_height))
    return env, avg_height


if __name__ == "__main__":
    stack()