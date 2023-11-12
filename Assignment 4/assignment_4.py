import numpy as np
import time
import os
try:
    import open3d as o3d
    visualize = True
except ImportError:
    print('To visualize you need to install Open3D. \n \t>> You can use "$ pip install open3d"')
    visualize = False

from assignment_4_helper import ICPVisualizer, load_point_cloud, view_point_cloud, quaternion_matrix, \
    quaternion_from_axis_angle, load_pcs_and_camera_poses, save_point_cloud
from scipy.spatial import KDTree

def transform_point_cloud(point_cloud, t, R):
    """
    Transform a point cloud applying a rotation and a translation
    :param point_cloud: np.arrays of size (N, 6)
    :param t: np.array of size (3,) representing a translation.
    :param R: np.array of size (3,3) representing a 3D rotation matrix.
    :return: np.array of size (N,6) resulting in applying the transformation (t,R) on the point cloud point_cloud.
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    point_cloud_xyz = point_cloud[:, :3] 
    point_cloud_rgb = point_cloud[:, 3:] 
    transform_point_cloud_xyz = (R @ point_cloud_xyz.T).T + t

    transformed_point_cloud = np.concatenate((transform_point_cloud_xyz, point_cloud_rgb), axis=1)  # TODO: Replace None with your result
    # ------------------------------------------------
    return transformed_point_cloud


def merge_point_clouds(point_clouds, camera_poses):
    """
    Register multiple point clouds into a common reference and merge them into a unique point cloud.
    :param point_clouds: List of np.arrays of size (N_i, 6)
    :param camera_poses: List of tuples (t_i, R_i) representing the camera i pose.
              - t: np.array of size (3,) representing a translation.
              - R: np.array of size (3,3) representing a 3D rotation matrix.
    :return: np.array of size (N, 6) where $$N = sum_{i=1}^K N_i$$
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    merged_point_cloud = None   # TODO: Replace None with your result

    for i in range(len(point_clouds)):
        point_cloud = np.array(point_clouds[i])
        t = np.array(camera_poses[i][0])
        R = np.array(camera_poses[i][1])
        transformed_point_cloud = transform_point_cloud(point_cloud, t, R)
        if merged_point_cloud is None:
            merged_point_cloud = transformed_point_cloud
        else:
            merged_point_cloud = np.concatenate((merged_point_cloud, transformed_point_cloud), axis=0)
    # ------------------------------------------------
    return merged_point_cloud


def find_closest_points(point_cloud_A, point_cloud_B):
    """
    Find the closest point in point_cloud_B for each element in point_cloud_A.
    :param point_cloud_A: np.array of size (n_a, 6)
    :param point_cloud_B: np.array of size (n_b, 6)
    :return: np.array of size(n_a,) containing the closest point indexes in point_cloud_B
            for each point in point_cloud_A
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    tree = KDTree(point_cloud_B[:, :3])
    _, closest_points_indxs = tree.query(point_cloud_A[:, :3], k=1) # TODO: Replace None with your result
    # ------------------------------------------------
    return closest_points_indxs


def find_best_transform(point_cloud_A, point_cloud_B):
    """
    Find the transformation 2 corresponded point clouds.
    Note 1: We assume that each point in the point_cloud_A is corresponded to the point in point_cloud_B at the same location.
        i.e. point_cloud_A[i] is corresponded to point_cloud_B[i] forall 0<=i<N
    :param point_cloud_A: np.array of size (N, 6) (scene)
    :param point_cloud_B: np.array of size (N, 6) (model)
    :return:
         - t: np.array of size (3,) representing a translation between point_cloud_A and point_cloud_B
         - R: np.array of size (3,3) representing a 3D rotation between point_cloud_A and point_cloud_B
    Note 2: We transform the model to match the scene.
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    points_A = point_cloud_A[:, :3]
    points_B = point_cloud_B[:, :3]
    
    # Compute centroids of both point clouds
    mu_A = np.mean(points_A, axis=0)
    mu_B = np.mean(points_B, axis=0)
    
    # Center the points by subtracting centroids
    centered_points_A = points_A - mu_A
    centered_points_B = points_B - mu_B

    # Compute the covariance matrix W
    W = np.dot(centered_points_A.T, centered_points_B)

    # Perform Singular Value Decomposition (SVD) on W
    U, _, Vt = np.linalg.svd(W)
    
    R = U @ Vt    # TODO: Replace None with your result
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    t = mu_A - np.dot(R, mu_B)    # TODO: Replace None with your result
    # ------------------------------------------------
    return t, R


def icp_step(point_cloud_A, point_cloud_B, t_init, R_init):
    """
    Perform an ICP iteration to find a new estimate of the pose of the model point cloud with respect to the scene pointcloud.
    :param point_cloud_A: np.array of size (N_a, 6) (scene)
    :param point_cloud_B: np.array of size (N_b, 6) (model)
    :param t_init: np.array of size (3,) representing the initial transformation candidate
                    * It may be the output from the previous iteration
    :param R_init: np.array of size (3,3) representing the initial rotation candidate
                    * It may be the output from the previous iteration
    :return:
        - t: np.array of size (3,) representing a translation estimate between point_cloud_A and point_cloud_B
        - R: np.array of size (3,3) representing a 3D rotation estimate between point_cloud_A and point_cloud_B
        - correspondences: np.array of size(n_a,) containing the closest point indexes in point_cloud_B
            for each point in point_cloud_A
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    # t = None    # TODO: Replace None with your result
    # R = None    # TODO: Replace None with your result
    # correspondences = None  # TODO: Replace None with your result

    transformed_point_cloud_B = transform_point_cloud(point_cloud_B, t_init, R_init)
    correspondences = find_closest_points(transformed_point_cloud_B, point_cloud_A)
    t, R = find_best_transform(point_cloud_A[correspondences], point_cloud_B)

    # correspondences = find_closest_points(point_cloud_A, transformed_point_cloud_B)
    # corresponding_points_B = point_cloud_B[correspondences]
    # t, R = find_best_transform(point_cloud_A[:, :6], corresponding_points_B)
    # ------------------------------------------------
    return t, R, correspondences


def icp(point_cloud_A, point_cloud_B, num_iterations=50, t_init=None, R_init=None, visualize=True):
    """
    Find the
    :param point_cloud_A: np.array of size (N_a, 6) (scene)
    :param point_cloud_B: np.array of size (N_b, 6) (model)
    :param num_iterations: <int> number of icp iteration to be performed
    :param t_init: np.array of size (3,) representing the initial transformation candidate
    :param R_init: np.array of size (3,3) representing the initial rotation candidate
    :param visualize: <bool> Whether to visualize the result
    :return:
         - t: np.array of size (3,) representing a translation estimate between point_cloud_A and point_cloud_B
         - R: np.array of size (3,3) representing a 3D rotation estimate between point_cloud_A and point_cloud_B
    """
    if t_init is None:
        t_init = np.zeros(3)
    if R_init is None:
        R_init = np.eye(3)
    if visualize:
        vis = ICPVisualizer(point_cloud_A, point_cloud_B)
    t = t_init
    R = R_init
    correspondences = None  # Initialization waiting for a value to be assigned
    if visualize:
        vis.view_icp(R=R, t=t)
        # vis.view_icp(R=R, t=t, frame_id=100)
        # for _ in range(150):  # adjust as desired
        #     time.sleep(0.1)
        #     vis.vis.poll_events()
        #     vis.vis.update_renderer()
    for i in range(num_iterations):
        # ------------------------------------------------
        # # FILL WITH YOUR CODE
        # t = None    # TODO: Replace None with your result
        # R = None  # TODO: Replace None with your result
        # correspondences = None  # TODO: Replace None with your result

        t, R, correspondences = icp_step(point_cloud_A, point_cloud_B, t, R)
        # ------------------------------------------------
        if visualize:
            vis.plot_correspondences(correspondences)   # Visualize point correspondences
            time.sleep(.5)  # Wait so we can visualize the correspondences
            vis.view_icp(R, t)  # Visualize icp iteration

    return t, R


def filter_point_cloud(point_cloud):
    """
    Remove unnecessary point given the scene point_cloud.
    :param point_cloud: np.array of size (N,6)
    :return: np.array of size (n,6) where n <= N
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    # filtered_pc = point_cloud  # TODO: Replace with your result

    # Define the bounding box for the object
    min_x, max_x = -0.6, 0.6
    min_y, max_y = -0.6, 0.6

    # Filter points based on their positions within the bounding box
    position_filtered_pc = point_cloud[(point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x) &
                                       (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y)]

    # Define the color threshold for the object
    yellow_lower_bound = [0.4, 0.4, 0.0]  # Lower bound for yellow
    yellow_upper_bound = [1.0, 1.0, 0.43]  # Upper bound for yellow

    # Filter points based on color (RGB values)
    filtered_pc = position_filtered_pc[
        ((position_filtered_pc[:, 3] >= yellow_lower_bound[0]) & (position_filtered_pc[:, 3] <= yellow_upper_bound[0])) &
        ((position_filtered_pc[:, 4] >= yellow_lower_bound[1]) & (position_filtered_pc[:, 4] <= yellow_upper_bound[1])) &
        ((position_filtered_pc[:, 5] >= yellow_lower_bound[2]) & (position_filtered_pc[:, 5] <= yellow_upper_bound[2]))
    ]

    # ------------------------------------------------
    return filtered_pc


def custom_icp(point_cloud_A, point_cloud_B, num_iterations=50, t_init=None, R_init=None, visualize=True):
    """
        Find the
        :param point_cloud_A: np.array of size (N_a, 6) (scene)
        :param point_cloud_B: np.array of size (N_b, 6) (model)
        :param num_iterations: <int> number of icp iteration to be performed
        :param t_init: np.array of size (3,) representing the initial transformation candidate
        :param R_init: np.array of size (3,3) representing the initial rotation candidate
        :param visualize: <bool> Whether to visualize the result
        :return:
             - t: np.array of size (3,) representing a translation estimate between point_cloud_A and point_cloud_B
             - R: np.array of size (3,3) representing a 3D rotation estimate between point_cloud_A and point_cloud_B
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE (OPTIONAL)
    filtered_point_cloud_A = filter_point_cloud(point_cloud_A)
    t, R = icp(filtered_point_cloud_A, point_cloud_B, num_iterations=num_iterations, t_init=t_init, R_init=R_init, visualize=visualize)  #TODO: Edit as needed (optional)
    # ------------------------------------------------
    return t, R



# ===========================================================================

# Test functions:

def transform_point_cloud_example(path_to_pointcloud_files, visualize=True):
    pc_source = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Source
    pc_goal = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med_tr.ply'))  # Transformed Goal
    t_gth = np.array([-0.5, 0.5, -0.2])
    r_angle = np.pi / 3
    R_gth = quaternion_matrix(np.array([np.cos(r_angle / 2), 0, np.sin(r_angle / 2), 0]))
    pc_tr = transform_point_cloud(pc_source, t=t_gth, R=R_gth)  # Apply your transformation to the source point cloud
    # Paint the transformed in red
    pc_tr[:, 3:] = np.array([.73, .21, .1]) * np.ones((pc_tr.shape[0], 3))  # Paint it red
    if visualize:
        # Visualize first without transformation
        print('Printing the source and goal point clouds')
        view_point_cloud([pc_source, pc_goal])
        # Visualize the transformation
        print('Printing the transformed output (in red) along source and goal point clouds')
        view_point_cloud([pc_source, pc_goal, pc_tr])
    else:
        # Save the pc so we can visualize them using other software
        save_point_cloud(np.concatenate([pc_source, pc_goal], axis=0), 'tr_pc_example_no_transformation',
                     path_to_pointcloud_files)
        save_point_cloud(np.concatenate([pc_source, pc_goal, pc_tr], axis=0), 'tr_pc_example_transform_applied',
                     path_to_pointcloud_files)
        print('Transformed point clouds saved as we cannot visualize them.\n Use software such as Meshlab to visualize them.')


def reconstruct_scene(path_to_pointcloud_files, visualize=True):
    pcs, camera_poses = load_pcs_and_camera_poses(path_to_pointcloud_files)
    pc_reconstructed = merge_point_clouds(pcs, camera_poses)
    if visualize:
        print('Displaying reconstructed point cloud scene.')
        view_point_cloud(pc_reconstructed)
    else:
        print(
            'Reconstructed scene point clouds saved as we cannot visualize it.\n Use software such as Meshlab to visualize them.')
        save_point_cloud(pc_reconstructed, 'reconstructed_scene_pc', path_to_pointcloud_files)


def perfect_model_icp(path_to_pointcloud_files, visualize=True):
    # Load the model
    pcB = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Model
    pcB[:, 3:] = np.array([.73, .21, .1]) * np.ones((pcB.shape[0], 3))  # Paint it red
    pcA = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Perfect scene
    # Apply transfomation to scene so they differ
    t_gth = np.array([0.4, -0.2, 0.2])
    r_angle = np.pi / 2
    R_gth = quaternion_matrix(np.array([np.cos(r_angle / 2), 0, np.sin(r_angle / 2), 0]))
    pcA = transform_point_cloud(pcA, R=R_gth, t=t_gth)
    R_init = np.eye(3)
    t_init = np.mean(pcA[:, :3], axis=0)

    # ICP -----
    t, R = icp(pcA, pcB, num_iterations=70, t_init=t_init, R_init=R_init, visualize=visualize)
    print('Infered Position: ', t)
    print('Infered Orientation:', R)
    print('\tReal Position: ', t_gth)
    print('\tReal Orientation:', R_gth)


def real_model_icp(path_to_pointcloud_files, visualize=True):
    # Load the model
    pcB = load_point_cloud(os.path.join(path_to_pointcloud_files, 'michigan_M_med.ply'))  # Model
    pcB[:, 3:] = np.array([.73, .21, .1]) * np.ones((pcB.shape[0], 3)) # Paint it red
    # ------ Noisy partial view scene -----
    pcs, camera_poses = load_pcs_and_camera_poses(path_to_pointcloud_files)
    pc = merge_point_clouds(pcs, camera_poses)
    pcA = filter_point_cloud(pc)
    if visualize:
        print('Displaying filtered point cloud. Close the window to continue.')
        view_point_cloud(pcA)
    else:
        print('Filtered scene point clouds saved as we cannot visualize it.\n Use software such as Meshlab to visualize them.')
        save_point_cloud(pcA, 'filtered_scene_pc', path_to_pointcloud_files)
    R_init = quaternion_matrix(quaternion_from_axis_angle(axis=np.array([0, 0, 1]), angle=np.pi / 2))
    t_init = np.mean(pcA[:, :3], axis=0)
    t_init[-1] = 0
    t, R = custom_icp(pcA, pcB, num_iterations=70, t_init=t_init, R_init=R_init)
    print('Infered Position: ', t)
    print('Infered Orientation:', R)


if __name__ == '__main__':
    # by default we assume that the point cloud files are on the same directory

    path_to_files = './a4_pointcloud_files' # TODO: Change the path to the directory containing your point cloud files

    # Test for part 1
    # transform_point_cloud_example(path_to_files, visualize=visualize) # TODO: Uncomment to test

    # Test for part 2
    # reconstruct_scene(path_to_files, visualize=visualize) # TODO: Uncomment to test

    # Test for part 5
    # perfect_model_icp(path_to_files, visualize=visualize) # TODO: Uncomment to test

    # Test for part 6
    real_model_icp(path_to_files, visualize=visualize)    # TODO: Uncomment to test

