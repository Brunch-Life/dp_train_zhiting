import numpy as np
import os
import cv2
import time
from tqdm import *
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


if __name__ == "__main__":
    root_dir = ("/home/chenyinuo/data/bingwen/diffusion_policy/data/test_green_bell_pepper/bingwen/data_for_success/")
    save_dir =  "/home/chenyinuo/data/bingwen/diffusion_policy/data/test_green_bell_pepper_delta_bingwen_new_T"

    episode_idx = 0
    os.makedirs(save_dir, exist_ok=True)
    for task in os.listdir(root_dir):
        load_dir = os.path.join(root_dir, task, "success")
        for episode in tqdm(os.listdir(load_dir), total=len(os.listdir(load_dir))):
            try:
                # t0 = time.time()
                data = np.load(os.path.join(load_dir, episode),allow_pickle=True,)["arr_0"].tolist()
                # print("Load time (without tolist):", time.time() - t0)
            except:
                continue

            # get state
            state_pos, state_quat = data["state"]["end"]["position"].squeeze(), data["state"]["end"]["orientation"].squeeze()

            state_tcp_pose = np.concatenate([state_pos, state_quat],axis=-1,) # (T, 7)
            gripper_width = data["state"]["effector"]["position_gripper"].squeeze()
            gripper_width = (gripper_width - (-0.01))/(0.04 - (-0.01))

            # get action
            action_gripper = data["action"]["effector"]["position_gripper"] # shape (T, 1)
            action_pos, action_quat = data["action"]["end"]["position"].squeeze(), data["action"]["end"]["orientation"].squeeze()
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
            delta_action_output = np.concatenate([delta_action_pos, delta_action_euler, action_gripper,],axis=-1).squeeze() # shape (T, 7)

            # # others test in base frame
            # delta_action_pos = np.zeros_like(data["action"]["end"]["position"].squeeze())
            # delta_action_pos[0] = data["action"]["end"]["position"][0].squeeze() - data["state"]["end"]["position"][0].squeeze()
            # delta_action_pos[1:] = data["action"]["end"]["position"][1:].squeeze() - data["action"]["end"]["position"][:-1].squeeze()

            # delta_action_euler = np.zeros_like(data["action"]["end"]["position"].squeeze())
            # delta_action_euler[0] = get_delta_euler_angles(
            #     quat_t1 = data["action"]["end"]["orientation"][0].squeeze(), 
            #     quat_t0 = data["state"]["end"]["orientation"][0].squeeze(),
            #     frame=frame,
            # )
            # delta_action_euler[1:] = get_seq_delta_euler_angles(
            #     seq_t1=data["action"]["end"]["orientation"][1:].squeeze(), 
            #     seq_t0=data["action"]["end"]["orientation"][:-1].squeeze(),
            #     frame=frame,
            # )
            # temp_delta_action = np.concatenate([delta_action_pos, delta_action_euler, action_gripper,],axis=-1).squeeze()

            save_path = os.path.join(
                save_dir,
                f"seed_{episode_idx}",
                f"ep_0",
            )
            os.makedirs(save_path, exist_ok=True)
            dict_data = {
                "tcp_pose": state_tcp_pose,           # shape: (T, 3+4)
                "gripper_width": gripper_width,       # shape: (T,)
                "delta_action": delta_action_output,  # shape: (T, 3+3+1)
                "abs_action": abs_action_output,      # shape: (T, 3+3+1)
            }
            np.save(os.path.join(save_path, "total_steps.npy"), dict_data)

            # np.savez_compressed(
            #     os.path.join(save_path, "total_steps.npz"),
            #     tcp_pose=state_tcp_pose, # state (T, 3+4)
            #     gripper_width=gripper_width, # state (T)
            #     action=delta_action_output, # delta_action_output or abs_action_output? (T, 3+3+1)
            # )

            # # for spped up the saving process
            # rgb_array = data["observation"]["rgb"]
            # bgr_array = rgb_array[..., ::-1]  # RGB to BGR
            # def save_image(time):
            #     bgr = bgr_array[time]
            #     filename = os.path.join(save_path, f"step_{time}_cam_third.jpg")
            #     cv2.imwrite(filename, bgr)
            # with ThreadPoolExecutor(max_workers=32) as executor:
            #     executor.map(save_image, range(data["observation"]["rgb"].shape[0]))

            for time in range(data["observation"]["rgb"].shape[0]):
                cv2.imwrite(
                    os.path.join(save_path, f"step_{time}_cam_third.jpg"),
                    cv2.cvtColor(data["observation"]["rgb"][time], cv2.COLOR_RGB2BGR),
                )
            episode_idx += 1
        break
