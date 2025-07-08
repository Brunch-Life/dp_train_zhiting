import numpy as np
import os
import cv2
from tqdm import *
from scipy.spatial.transform import Rotation as R


def get_pose(position, orientation):
    rotation_matrix = R.from_quat(orientation.reshape(-1)[[1, 2, 3, 0]]).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = position
    return pose


def seq_get_pose(position: np.ndarray, orientation: np.ndarray):
    seq_pose = []
    for x in range(position.shape[0]):
        seq_pose.append(get_pose(position[x], orientation[x]))
    seq_pose = np.stack(seq_pose)
    return seq_pose


def get_pos_and_ori(pose):
    position = pose[:3, 3]
    rotation_matrix = pose[:3, :3]
    orientation = R.from_matrix(rotation_matrix).as_quat()[[3, 0, 1, 2]]
    return position, orientation


def get_seq_pos_and_ori(seq_pose):
    seq_pos = []
    seq_ori = []
    for x in range(seq_pose.shape[0]):
        pos, ori = get_pos_and_ori(seq_pose[x])
        seq_pos.append(pos)
        seq_ori.append(ori)
    seq_pos = np.stack(seq_pos)
    seq_ori = np.stack(seq_ori)
    return seq_pos, seq_ori


def get_euler_angles(orientation):
    r = R.from_quat(orientation.reshape(-1)[[1, 2, 3, 0]])
    euler_angles = r.as_euler("xyz", degrees=False)
    return euler_angles


def get_seq_euler_angles(seq_orientation):
    seq_euler_angles = []
    for x in range(seq_orientation.shape[0]):
        seq_euler_angles.append(get_euler_angles(seq_orientation[x]))
    seq_euler_angles = np.stack(seq_euler_angles)
    return seq_euler_angles.reshape(-1, 1, 3)


root_dir = (
    "/iag_ad_01/ad/tangyinzhou/tyz/test_peach_plate/bingwen/data_for_success/peach_plate_wooden"
)

save_dir = "/iag_ad_01/ad/tangyinzhou/tyz/reward_diffusion_policy_relative/diffusion_policy/data/pickplace_0606"
episode_idx = 0
os.makedirs(save_dir, exist_ok=True)
for task in os.listdir(root_dir):
    load_dir = os.path.join(root_dir, task, "success")
    load_dir = "/iag_ad_01/ad/tangyinzhou/tyz/test_peach_plate/bingwen/data_for_success/peach_plate_wooden/success"
    for episode in tqdm(os.listdir(load_dir), total=len(os.listdir(load_dir))):
        try:
            data = np.load(
                os.path.join(load_dir, episode),
                allow_pickle=True,
            )["arr_0"].tolist()
        except:
            continue
        tcp_pose = np.concatenate(
            (data["state"]["end"]["position"], data["state"]["end"]["orientation"]),
            axis=2,
        )
        gripper_width = data["state"]["effector"]["position_gripper"].reshape(-1)
        robot_joints = data["state"]["joint"]["position"]
        # get world tcp_pose
        robot_pose = seq_get_pose(
            data["state"]["robot"]["position"], data["state"]["robot"]["orientation"]
        )
        end_pose = seq_get_pose(
            data["state"]["end"]["position"], data["state"]["end"]["orientation"]
        )
        world_tcp_pose = robot_pose * end_pose
        pos_seq, ori_seq = get_seq_pos_and_ori(world_tcp_pose)
        privileged_obs = np.concatenate(
            (pos_seq, ori_seq, gripper_width.reshape(-1, 1)), axis=1
        )

        from scipy.spatial.transform import Rotation as R

        # 将欧拉角转换为旋转矩阵
        def euler_to_rotation_matrix(euler_angles):
            return R.from_euler("xyz", euler_angles, degrees=False).as_matrix()

        def get_relative_euler_angles(ori1, ori2):
            # 将欧拉角转换为旋转矩阵
            ori1_matrix = euler_to_rotation_matrix(ori1.reshape(-1))
            ori2_matrix = euler_to_rotation_matrix(ori2.reshape(-1))

            # 计算相对旋转矩
            relative_rotation_matrix = np.dot(ori2_matrix, ori1_matrix.T)

            # 将相对旋转矩阵转换回欧拉角
            relative_euler_angles = R.from_matrix(relative_rotation_matrix).as_euler(
                "xyz", degrees=False
            )
            return relative_euler_angles

        def get_seq_relative_euler_angles(seq1, seq2):
            relative_euler_seq = []
            for x in range(seq1.shape[0]):
                relative_euler_seq.append(get_relative_euler_angles(seq1[x], seq2[x]))
            relative_euler_seq = np.stack(relative_euler_seq)
            return relative_euler_seq.reshape(-1, 1, 3)

        action_eular_angles = get_seq_euler_angles(data["action"]["end"]["orientation"])
        state_eular_angles = get_seq_euler_angles(data["state"]["end"]["orientation"])
        action = np.concatenate(
            [
                data["action"]["end"]["position"] - data["state"]["end"]["position"],
                get_seq_relative_euler_angles(
                    action_eular_angles,
                    state_eular_angles,
                ),
                data["action"]["effector"]["position_gripper"].reshape(
                    data["action"]["effector"]["position_gripper"].shape + (1,)
                ),
            ],
            axis=2,
        ).squeeze(1)
        desired_grasp_pose = np.concatenate(
            (data["action"]["end"]["position"], data["action"]["end"]["orientation"]),
            axis=2,
        ).squeeze(1)
        desired_gripper_width = data["action"]["effector"]["position_gripper"].reshape(
            -1
        )
        # save_file = {
        #     "tcp_pose": tcp_pose,
        #     "gripper_width": gripper_width,
        #     "robot_joints": robot_joints,
        #     "privileged_obs": privileged_obs,
        #     "action": action,
        #     "desired_grasp_pose": desired_grasp_pose,
        #     "desired_gripper_width": desired_gripper_width,
        # }
        save_path = os.path.join(
            save_dir,
            f"seed_{episode_idx}",
            f"ep_0",
        )
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_path, "total_steps.npz"),
            tcp_pose=tcp_pose.squeeze(1),
            gripper_width=gripper_width,
            robot_joints=robot_joints,
            privileged_obs=privileged_obs,
            action=action,
            desired_grasp_pose=desired_grasp_pose,
            desired_gripper_width=desired_gripper_width,
        )
        for time in range(privileged_obs.shape[0]):
            cv2.imwrite(
                os.path.join(save_path, f"step_{time}_cam_third.jpg"),
                cv2.cvtColor(data["observation"]["rgb"][time], cv2.COLOR_RGB2BGR),
            )
        episode_idx += 1
    break
