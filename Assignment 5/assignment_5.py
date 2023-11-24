import numpy as np
import pybullet as p
import open3d as o3d
import assignment_5_helper as helper


def get_antipodal(pcd):
    """
    function to compute antipodal grasp given point cloud pcd
    :param pcd: point cloud in open3d format (converted to numpy below)
    :return: gripper pose (4, ) numpy array of gripper pose (x, y, z, theta)
    """
    # convert pcd to numpy arrays of points and normals
    pc_points = np.asarray(pcd.points)
    pc_normals = np.asarray(pcd.normals)

    # ------------------------------------------------
    # FILL WITH YOUR CODE
    gripper_width_max=0.15
    normal_alignment_threshold=0
    voxel_size = 0.02
    # gripper orientation - replace 0. with your calculations
    theta = 0.
    # gripper pose: (x, y, z, theta) - replace 0. with your calculations
    gripper_pose = np.array([0., 0., 0., theta])

    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    pc_points = np.asarray(downsampled_pcd.points)
    pc_normals = np.asarray(downsampled_pcd.normals)

    # points = downsampled_points.T
    # kdtree = o3d.geometry.KDTreeFlann(downsampled_pcd)
    # _, p2_ids, _ = kdtree.search_knn_vector_3d(points, knn)  # Use 2 neighbors for each point

    # gripper_directions = downsampled_points[p2_ids[:, 1]] - downsampled_points[p2_ids[:, 0]]
    # thetas = np.arctan2(gripper_directions[:, 1], gripper_directions[:, 0])
    # normal_dot_products = np.sum(downsampled_normals[p2_ids[:, 0]] * downsampled_normals[p2_ids[:, 1]], axis=1)
    # valid_indices = np.where(
    #     (np.linalg.norm(gripper_directions, axis=1) <= gripper_width_max) &
    #     (normal_dot_products < normal_alignment_threshold)
    # )

    # if valid_indices[0].size > 0:
    #     best_pair_index = valid_indices[0][0]  # Select the first valid pair
    #     p1 = downsampled_points[p2_ids[best_pair_index, 0]]
    #     theta = thetas[best_pair_index]
    #     gripper_pose = np.array([p1[0], p1[1], p1[2], theta])

    distances = np.linalg.norm(pc_points[:, np.newaxis, :] - pc_points[np.newaxis, :, :], axis=2)
    valid_pairs = distances <= gripper_width_max
    valid_indices = np.where(valid_pairs)

    normals_dot_product = np.einsum('ij,ij->i', pc_normals[valid_indices[0]], pc_normals[valid_indices[1]])
    valid_normals = normals_dot_product < normal_alignment_threshold
    final_pairs = np.where(valid_normals)[0]
    
    if final_pairs.size > 0:
        best_pair_index = np.random.choice(final_pairs)
        p1, p2 = pc_points[valid_indices[0][best_pair_index]], pc_points[valid_indices[1][best_pair_index]]
        midpoint = (p1 + p2) / 2
        gripper_direction = p2 - p1
        theta = np.arctan2(gripper_direction[1], gripper_direction[0])
        gripper_pose[:3] = midpoint
        gripper_pose[-1] = theta
    # ------------------------------------------------

    return gripper_pose


def main(n_tries=5):
    # Initialize the world
    world = helper.World()

    # start grasping loop
    # number of tries for grasping
    for i in range(n_tries):
        # get point cloud from cameras in the world
        pcd = world.get_point_cloud()
        # check point cloud to see if there are still objects to remove
        finish_flag = helper.check_pc(pcd)
        if finish_flag:  # if no more objects -- done!
            print('===============')
            print('Scene cleared')
            print('===============')
            break
        # visualize the point cloud from the scene
        helper.draw_pc(pcd)
        # compute antipodal grasp
        gripper_pose = get_antipodal(pcd)
        # send command to robot to execute
        robot_command = world.grasp(gripper_pose)
        # robot drops object to the side
        world.drop_in_bin(robot_command)
        # robot goes to initial configuration and prepares for next grasp
        world.home_arm()
        # go back to the top!

    # terminate simulation environment once you're done!
    p.disconnect()
    return finish_flag


if __name__ == "__main__":
    flag = main()
