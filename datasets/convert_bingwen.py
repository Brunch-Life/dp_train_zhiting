import os
import cv2
import time
import tyro
import shutil
import numpy as np
from tqdm import *
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from transforms3d.euler import euler2mat, mat2euler, quat2euler, quat2mat, euler2quat
from utils.math_utils import get_pose_from_rot_pos
from concurrent.futures import ThreadPoolExecutor


def get_delta_T_matrix(action_t0, action_t1, frame: str):
    T_t0 = get_pose_from_rot_pos(quat2mat(action_t0[3:]), action_t0[:3])
    T_t1 = get_pose_from_rot_pos(quat2mat(action_t1[3:]), action_t1[:3])
    if frame == "base":
        delta_T_matrix = T_t1 @ np.linalg.inv(T_t0)
    elif frame == "tool":
        delta_T_matrix = np.linalg.inv(T_t0) @ T_t1
    else:
        raise ValueError("Frame must be either 'base' or 'tool'.")
    return delta_T_matrix

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


def get_seq_delta_euler_angles(seq_t0, seq_t1, frame: str = "tool"): # we prefer to use tool frame
    delta_euler_seq = []
    for i in range(seq_t0.shape[0]):
        delta_euler_seq.append(get_delta_euler_angles(quat_t0=seq_t0[i], quat_t1=seq_t1[i], frame=frame))
    delta_euler_seq = np.array(delta_euler_seq)
    return delta_euler_seq

# clean diffuser
def get_seq_delta_T_matrix(seq_t0, seq_t1, frame: str = "tool"): # we prefer to use tool frame
    delta_T_seq = []
    for i in range(seq_t0.shape[0]):
        delta_T_seq.append(get_delta_T_matrix(action_t0=seq_t0[i], action_t1=seq_t1[i], frame=frame))
    delta_T_seq = np.array(delta_T_seq)
    return delta_T_seq


def get_pos_euler_from_T_matrix(T_matrix): # only for test
    position = T_matrix[:3, 3]
    rot_matrix = T_matrix[:3, :3]
    euler = mat2euler(rot_matrix)
    return position, euler

# not use now
def inv_scale_action(action, low, high):
    """Inverse of `clip_and_scale_action` without clipping. -> [-1, 1]"""
    return (action - 0.5 * (high + low)) / (0.5 * (high - low))


def squeeze_dict(data):
    data["state"]["end"]["position"] = data["state"]["end"]["position"].squeeze()
    data["state"]["end"]["orientation"] = data["state"]["end"]["orientation"].squeeze()
    data["state"]["effector"]["position_gripper"] = data["state"]["effector"]["position_gripper"].squeeze()
    data["action"]["end"]["position"] = data["action"]["end"]["position"].squeeze()
    data["action"]["end"]["orientation"] = data["action"]["end"]["orientation"].squeeze()
    data["action"]["effector"]["position_gripper"] = data["action"]["effector"]["position_gripper"].squeeze()
    return data

@dataclass
class Args:
    root_dir: str
    save_dir: str
    max_task_num: Optional[int] = None

if __name__ == "__main__":
    args = tyro.cli(Args)
    root_dir = (args.root_dir)
    save_dir = args.save_dir
    print(f"Root directory: {root_dir}")
    print(f"Save directory: {save_dir}")

    episode_idx = 0
    os.makedirs(save_dir, exist_ok=True)
    all_tasks = os.listdir(root_dir)
    for task_idx, task in enumerate(all_tasks, start=1):
        if args.max_task_num is not None and task_idx > args.max_task_num:
            print(f"Reached max task limit: {args.max_task_num}. Stopping.")
            break
        load_dir = os.path.join(root_dir, task, "success")
        print(f"\n🔧 Processing task {task_idx}/{len(all_tasks)}: {task}")
        print(f"📂 Loading from: {load_dir}")
        episodes = os.listdir(load_dir)
        for episode in tqdm(episodes, desc=f"Task {task_idx}: {task}", total=len(episodes)):
            try:
                if episode.endswith(".npz"):
                    data = np.load(os.path.join(load_dir, episode), allow_pickle=True)["arr_0"]
                    data = data.tolist()
                elif episode.endswith(".npy"):
                    data = np.load(os.path.join(load_dir, episode), allow_pickle=True).item()
                else:
                    print(f"Skipping {episode}, not a valid file format.")
                    continue
            except Exception as e:
                print(f"Error loading {episode}. Skipping...")
                print(e)
                continue
            
            # squeeze the data
            data = squeeze_dict(data)

            # get state
            state_pos, state_quat = data["state"]["end"]["position"], data["state"]["end"]["orientation"]

            state_tcp_pose = np.concatenate([state_pos, state_quat],axis=-1,) # (T, 7)
            state_gripper_width = data["state"]["effector"]["position_gripper"]

            # get action
            action_gripper = data["action"]["effector"]["position_gripper"][:,None] # shape (T, 1)
            action_pos, action_quat = data["action"]["end"]["position"], data["action"]["end"]["orientation"]
            action_euler = np.array([quat2euler(action_quat[i], "sxyz") for i in range(action_quat.shape[0])]) # shape (T, 3)
            abs_action_output = np.concatenate([action_pos, action_euler, action_gripper],axis=-1,) # shape (T, 7)

            frame = "tool" # or "tool"
            abs_action_temp = np.concatenate([action_pos, action_quat],axis=-1,)
            delta_T_matrix= np.zeros((abs_action_temp.shape[0], 4, 4), dtype=np.float32)
            delta_T_matrix[0] = get_delta_T_matrix(action_t0=state_tcp_pose[0], action_t1=abs_action_temp[0], frame=frame)
            delta_T_matrix[1:] = get_seq_delta_T_matrix(
                seq_t1=abs_action_temp[1:], 
                seq_t0=abs_action_temp[:-1],
                frame=frame,
            )
            delta_action_pos = delta_T_matrix[:, :3, 3]  # Extract position from the delta T matrix
            delta_action_mat = delta_T_matrix[:, :3, :3]  # Extract rotation matrix from the delta T matrix
            delta_action_euler = np.array([mat2euler(delta_action_mat[i]) for i in range(delta_action_mat.shape[0])])

            # action in `frame` frame
            delta_action_output = np.concatenate([delta_action_pos, delta_action_euler, action_gripper,],axis=-1) # shape (T, 7)

            save_path = os.path.join(
                save_dir,
                f"seed_{episode_idx}",
                f"ep_0",
            )
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
                print(f"Removed existing directory: {save_path}")
            os.makedirs(save_path, exist_ok=True)
            
            rgb_list = np.array(data["observation"]["rgb"], dtype=object) # for ragged data
            # wrist_rgb_list = np.array(data["observation"]["wrist_rgb"], dtype=object) # for ragged data
            wrist_rgb_list = None  # not used in this dataset
            
            np.savez_compressed(
                os.path.join(save_path, "total_steps.npz"),
                is_image_encode=data["observation"]["is_image_encode"],# bool
                tcp_pose=state_tcp_pose,                # state (T, 3+4)
                state_gripper_width=state_gripper_width, # state (T,)
                delta_action=delta_action_output,       # delta_action_output or abs_action_output? (T, 3+3+1)
                abs_action=abs_action_output,           # abs_action_output (T, 3+3+1)
                cam_third=rgb_list,                     # cam_third (T, H, W, 3)
                cam_wrist=wrist_rgb_list,               # cam_wrist (T, H, W, 3)
            )

            episode_idx += 1

