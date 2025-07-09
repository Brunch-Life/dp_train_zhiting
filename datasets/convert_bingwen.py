import numpy as np
import os
import cv2
from tqdm import *
from transforms3d.euler import euler2mat, mat2euler, quat2euler, quat2mat, euler2quat
from utils.math_utils import get_pose_from_rot_pos


def get_delta_T_matrix(T_t0, T_t1, frame: str):
    if frame == "base":
        delta_T_matrix = T_t1 @ np.linalg.inv(T_t0)
    elif frame == "tool":
        delta_T_matrix = np.linalg.inv(T_t0) @ T_t1
    else:
        raise ValueError("Frame must be either 'base' or 'tool'.")
    return delta_T_matrix

def get_pos_euler_from_T_matrix(T_matrix):
    position = T_matrix[:3, 3]
    rot_matrix = T_matrix[:3, :3]
    euler = mat2euler(rot_matrix)
    return position, euler

def get_delta_euler_angles(quat_t0, quat_t1, frame: str):
    matrix_t0 = quat2mat(quat_t0)
    matrix_t1 = quat2mat(quat_t1)
    if frame == "base":
        delta_matrix = matrix_t1 @ matrix_t0.T
    elif frame == "tool":
        delta_matrix = matrix_t0.T @ matrix_t1
    else:
        raise ValueError("Frame must be either 'base' or 'tool'.")
    return mat2euler(delta_matrix)


def get_seq_delta_euler_angles(seq_t0, seq_t1, frame: str = "base"):
    delta_euler_seq = []
    for i in range(seq_t0.shape[0]):
        delta_euler_seq.append(get_delta_euler_angles(quat_t0=seq_t0[i], quat_t1=seq_t1[i], frame=frame))
    delta_euler_seq = np.stack(delta_euler_seq)
    return delta_euler_seq.reshape(-1, 1, 3) # @ bingwen


def get_seq_delta_T_matrix(seq_t0, seq_t1, frame: str = "base"):
    delta_T_seq = []
    for i in range(seq_t0.shape[0]):
        T_t0 = get_pose_from_rot_pos(quat2mat(seq_t0[i]), seq_t0[i, :3])
        T_t1 = get_pose_from_rot_pos(quat2mat(seq_t1[i]), seq_t1[i, :3])
        delta_T_seq.append(get_delta_T_matrix(T_t0, T_t1, frame=frame))
    delta_T_seq = np.stack(delta_T_seq)
    return delta_T_seq



if __name__ == "__main__":
    root_dir = ("/home/chenyinuo/data/bingwen/diffusion_policy/data/test_green_bell_pepper/bingwen/data_for_success/")
    save_dir =  "/home/chenyinuo/data/bingwen/diffusion_policy/data/test_green_bell_pepper_delta_bingwen"

    episode_idx = 0
    os.makedirs(save_dir, exist_ok=True)
    for task in os.listdir(root_dir):
        load_dir = os.path.join(root_dir, task, "success")
        for episode in tqdm(os.listdir(load_dir), total=len(os.listdir(load_dir))):
            try:
                data = np.load(
                    os.path.join(load_dir, episode),
                    allow_pickle=True,
                )["arr_0"].tolist()
            except:
                continue

            # get state
            state_pos, state_quat = data["state"]["end"]["position"].squeeze(), data["state"]["end"]["orientation"].squeeze()

            state_tcp_pose = np.concatenate([state_pos, state_quat],axis=-1,)
            gripper_width = data["state"]["effector"]["position_gripper"].squeeze()
            gripper_width = (gripper_width - (-0.01))/(0.04 - (-0.01))

            # get action
            action_gripper = data["action"]["effector"]["position_gripper"].squeeze()
            action_pos, action_quat = data["action"]["end"]["position"].squeeze(), data["action"]["end"]["orientation"].squeeze()
            abs_action = np.concatenate([action_pos, quat2euler(action_quat, "sxyz")],axis=-1,)

            frame = "base" # or "tool"
            delta_action_pos = np.zeros_like(data["action"]["end"]["position"])
            delta_action_pos[0] = data["action"]["end"]["position"][0] - data["state"]["end"]["position"][0]
            delta_action_pos[1:] = data["action"]["end"]["position"][1:] - data["action"]["end"]["position"][:-1]

            delta_action_euler = np.zeros_like(data["action"]["end"]["position"])
            delta_action_euler[0] = get_delta_euler_angles(
                quat_t1 = data["action"]["end"]["orientation"][0], 
                quat_t0 = data["state"]["end"]["orientation"][0],
                frame=frame,
            )
            delta_action_euler[1:] = get_seq_delta_euler_angles(
                seq_t1=data["action"]["end"]["orientation"][1:], 
                seq_t0=data["action"]["end"]["orientation"][:-1],
                frame=frame,
            )

            # action in base frame
            delta_action = np.concatenate([delta_action_pos, delta_action_euler, action_gripper,],axis=2,).squeeze()

            save_path = os.path.join(
                save_dir,
                f"seed_{episode_idx}",
                f"ep_0",
            )
            os.makedirs(save_path, exist_ok=True)
            np.savez_compressed(
                os.path.join(save_path, "total_steps.npz"),
                tcp_pose=state_tcp_pose, # state
                gripper_width=gripper_width, # state
                action=abs_action, # delta_action or abs_action?
            )
            for time in range(data["observation"]["rgb"].shape[0]):
                cv2.imwrite(
                    os.path.join(save_path, f"step_{time}_cam_third.jpg"),
                    cv2.cvtColor(data["observation"]["rgb"][time], cv2.COLOR_RGB2BGR),
                )
            episode_idx += 1
        break
