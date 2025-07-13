import numpy as np
import torch
import os
import cv2
import pickle
from time import time
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2mat, mat2euler
from collections import defaultdict
import numpy as np
import threading
from PIL import Image
from utils.math_utils import (
    wrap_to_pi,
    euler2quat,
    quat2euler,
    get_pose_from_rot_pos,
)
import copy



timing_stats = defaultdict(list)
timing_lock = threading.Lock()  
batch_counter = 0
print_interval = 10 

class TimingContext:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        with timing_lock:
            timing_stats[self.name].append(elapsed)


def print_timing_stats():
    print("\n===== 数据加载时间统计 =====")
    for operation, times in sorted(timing_stats.items()):
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"{operation:<30} - 平均: {avg_time*1000:.2f}ms, 最大: {max_time*1000:.2f}ms, 最小: {min_time*1000:.2f}ms, 调用次数: {len(times)}")
    print("=========================\n")


def get_image_sequence_from_bytes(image_bytes_list):
    images_list = []
    for image_bytes in image_bytes_list: # for the length of each item in the list not equal
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        images_list.append(image)
    return np.array(images_list)

def get_image_from_bytes(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


class Sim2SimEpisodeDatasetEff(Dataset):
    def __init__(
            self,
            data_roots,
            num_seeds,
            chunk_size,
            split="train",
            norm_stats_path=None,
            augment_images=True,
            use_desired_action=True,
            use_pre_img=False,   # for debug, compare with pre_img
            **kwargs
    ):
        super().__init__()
        with TimingContext("dataset_initialization"):  #time
            self.data_roots = data_roots
            self.camera_names = kwargs.get("camera_names", ["third", "wrist"])
            self.chunk_size = chunk_size
            self.augment_images = augment_images
            self.use_desired_action = use_desired_action
            self.transformations = None
            self.split = split
            self.use_pre_img = use_pre_img 
            
            self.episode_data = {}
            episode_list = []
            num_steps_list = []

            with TimingContext("total_steps loading"): #time
                print("*"*20 + "total_steps loading" + "*"*20)
                for data_dir_idx, data_root in enumerate(self.data_roots):
                    for seed in tqdm(range(num_seeds), desc=f"Loading from {data_root}", position=0):
                        seed_path = os.path.join(data_root, f"seed_{seed}") # seed here means episode_index
                        total_steps_path = os.path.join(seed_path, "ep_0", "total_steps.npy") 
                        if not os.path.exists(total_steps_path):
                            total_steps_path = os.path.join(seed_path, "ep_0", "total_steps.npz") 
                            if not os.path.exists(total_steps_path):
                                continue
                        
                        if total_steps_path.endswith(".npz"):
                            single_traj_data = np.load(total_steps_path, allow_pickle=True)
                        elif total_steps_path.endswith(".npy"):
                            # to test
                            single_traj_data = np.load(total_steps_path, allow_pickle=True).item()
                        else:
                            raise ValueError(f"Unsupported file format: {total_steps_path}")

                        traj_num_steps = single_traj_data['tcp_pose'].shape[0]

                        if traj_num_steps > 1:
                            ep_id = 0  # ep_id is always 0
                            item_index = (data_dir_idx, seed, ep_id)  
                            episode_list.append(item_index)
                            num_steps_list.append(traj_num_steps)
                            self.episode_data[item_index] = {
                                "is_image_encode": single_traj_data['is_image_encode'],
                                'tcp_pose': single_traj_data['tcp_pose'],
                                'state_gripper_width': single_traj_data['state_gripper_width'],
                                'delta_action': single_traj_data['delta_action'],
                                'abs_action': single_traj_data['abs_action'],
                                'cam_third': single_traj_data['cam_third'],
                                'cam_wrist': single_traj_data['cam_wrist'],
                            }
                            
            print("episode_list", len(episode_list))
            if split == "train": #TODO(bingwen) 0.99 -> 0.9
                self.episode_list = episode_list[: int(0.9 * len(episode_list))]
                self.num_steps_list = num_steps_list[: int(0.9 * len(episode_list))]
            else:
                self.episode_list = episode_list[int(0.9 * len(episode_list)) :]
                self.num_steps_list = num_steps_list[int(0.9 * len(episode_list)) :]

            self.cum_steps_list = np.cumsum(self.num_steps_list)
            print(split, len(self.episode_list), self.cum_steps_list[-1])

            if norm_stats_path is None:
                stats = self.compute_normalize_stats()
            else:
                with TimingContext("load_norm_stats"):
                    stats = pickle.load(open(norm_stats_path, "rb"))
            self.update_obs_normalize_params(stats)

            result_0 = self.__getitem__(0) 

    def __len__(self):
        return self.cum_steps_list[-1]

    def __getitem__(self, index: int):
        with TimingContext("getitem_total"):
            with TimingContext("get_unnormalized_item"):    
                result = self.get_unnormalized_item(index)

            robot_state = result["robot_state"]
            proprio_state = result["proprio_state"]
            is_pad = result["is_pad"]
            action_chunk = result["action"]
            robot_state = (
                robot_state - self.pose_gripper_mean
            ) / self.pose_gripper_scale
            action_chunk[~is_pad] = (
                action_chunk[~is_pad] - np.expand_dims(self.pose_gripper_mean, axis=0)
            ) / np.expand_dims(self.pose_gripper_scale, axis=0)
            proprio_state = (
                proprio_state - self.proprio_gripper_mean
            ) / self.proprio_gripper_scale

            result["robot_state"] = torch.from_numpy(robot_state) # actually not used during trainig time. 
            result["proprio_state"] = torch.from_numpy(proprio_state)
            result["action"] = torch.from_numpy(action_chunk)
            result["is_pad"] = torch.from_numpy(is_pad)
            # for debug 
            # result["pose_chunk"] = torch.from_numpy(result["pose_chunk"]) #

            images = torch.from_numpy(result["images"])
            images = torch.einsum('k h w c -> k c h w', images)

            if self.use_pre_img == False:
                with TimingContext("image_processing"):
                    if self.transformations is None:
                        # print('Initializing transformations')
                        original_size = images.shape[2:]
                        ratio = 0.95
                        self.transformations = [
                            transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                            transforms.Resize((224, 224), antialias=True),
                        ]

                    if self.augment_images:
                        for transform in self.transformations:
                            images = transform(images)
            else:
                assert images.shape[2] == 224 and images.shape[3] == 224, "processed image size should be 224x224"

            images = images / 255.0
            result["images"] = images

            return result

    def _locate(self, index):
        assert index < len(self)
        traj_idx = np.where(self.cum_steps_list > index)[0][0]
        steps_before = self.cum_steps_list[traj_idx - 1] if traj_idx > 0 else 0
        start_ts = index - steps_before
        return traj_idx, start_ts

    def get_unnormalized_item(self, index):
        result_dict = {}
        result_dict["lang"] = " "
        
        with TimingContext("locate_trajectory"):
            traj_idx, start_ts = self._locate(index)
            end_ts = min(self.num_steps_list[traj_idx], start_ts + self.chunk_size + 1)
            is_pad = np.zeros((self.chunk_size,), dtype=bool)
            if end_ts < start_ts + self.chunk_size + 1:
                is_pad[-(start_ts + self.chunk_size + 1 - end_ts):] = True
            result_dict["is_pad"] = is_pad

            data_dir_idx, seed, ep_id = self.episode_list[traj_idx]
            data_root = self.data_roots[data_dir_idx]

        with TimingContext("data_fetching"):
            episode_data = self.episode_data[(data_dir_idx, seed, ep_id)]
            
        # image
        with TimingContext("image_loading"):
            images = []

            for cam_name in self.camera_names:
                image = episode_data.get("cam_"+cam_name, None)[start_ts]
                if image is None:
                    continue

                if episode_data["is_image_encode"]:
                    try:
                        image = get_image_from_bytes(image)
                        # # save the jpeg, just for debug
                        # Image.fromarray(image).save(os.path.join(data_root, f"seed_{seed}", f"ep_{ep_id}", f"{cam_name}_img_{start_ts}.jpeg"))
                    except Exception as e:
                        import wandb
                        print(f"Error decoding image for {cam_name} at index {index}: {e}")
                        wandb.log({"error": f"Error decoding image for {cam_name} at index {index}: {e}"})
                        continue
                
                images.append(image)

            images = np.stack(images, axis=0)
            result_dict["images"] = images

        with TimingContext("pose_processing"):
            pose_at_obs = None
            abs_pose_chunk = []
            action_gripper_width_chunk = []
            proprio_state = np.zeros((10,), dtype=np.float32)
            robot_state = np.zeros((10,), dtype=np.float32)

            for step_idx in range(start_ts, end_ts): # in an action chunk(from start to end)
                tcp_pose = episode_data["tcp_pose"][step_idx]
                pose_p, pose_q = tcp_pose[:3], tcp_pose[3:]
                pose_mat = quat2mat(pose_q)
                obs_pose = get_pose_from_rot_pos(pose_mat, pose_p)

                if step_idx == start_ts:
                    pose_at_obs = obs_pose # chunk t=0
                    pose_mat_6 = pose_mat[:, :2].reshape(-1)
                    proprio_state[:] = np.concatenate(
                        [
                            pose_p,
                            pose_mat_6,
                            np.array([episode_data["state_gripper_width"][step_idx]]),
                        ]
                    )
                    robot_state[-1] = episode_data["state_gripper_width"][step_idx]

                elif step_idx > start_ts:
                    abs_action = episode_data["abs_action"][step_idx]
                    abs_action_pose = get_pose_from_rot_pos(
                        euler2mat(abs_action[3], abs_action[4], abs_action[5], "sxyz"), abs_action[:3]
                    )
                    abs_pose_chunk.append(abs_action_pose)
                    action_gripper_width_chunk.append(abs_action[-1:])

            # compute the relative pose
            _pose_relative = np.eye(4)
            robot_state[:9] = np.concatenate(
                [_pose_relative[:3, 3], _pose_relative[:3, :2].reshape(-1)]
            )

            delta_action_chunk = np.zeros((self.chunk_size, 10), dtype=np.float32)
            for i in range(end_ts - start_ts - 1): # only use delta action to train
                _pose_relative = np.linalg.inv(pose_at_obs) @ abs_pose_chunk[i] # in tool base
                delta_action_chunk[i] = np.concatenate(
                    [
                        _pose_relative[:3, 3],
                        _pose_relative[:3, :2].reshape(-1),
                        action_gripper_width_chunk[i],
                    ]
                )

            result_dict["robot_state"] = robot_state # the pose of robot in the world frame, shape (10,), not used during training
            result_dict["proprio_state"] = proprio_state # shape (10,)
            result_dict["action"] = delta_action_chunk # shape (chunk_size, 10)

        return result_dict

    def compute_normalize_stats(self, scale_eps=0.03):
        print("compute normalize stats...")
        # min and max scale
        joint_min, joint_max = None, None
        gripper_width_min, gripper_width_max = None, None
        pose_min, pose_max = None, None
        proprio_min, proprio_max = None, None

        def safe_minimum(a: np.ndarray, b: np.ndarray):
            if a is None:
                return b
            if b is None:
                return a
            return np.minimum(a, b)

        def safe_maximum(a: np.ndarray, b: np.ndarray):
            if a is None:
                return b
            if b is None:
                return a
            return np.maximum(a, b)

        def safe_min(a: np.ndarray, axis: int):
            if a.shape[axis] == 0:
                return None
            return np.min(a, axis=axis)

        def safe_max(a: np.ndarray, axis: int):
            if a.shape[axis] == 0:
                return None
            return np.max(a, axis=axis)

        for i in tqdm(range(len(self))):

            item_dict = self.get_unnormalized_item(i)
            pose = item_dict["robot_state"][:9] # not be used
            action_pose = item_dict["action"][~item_dict["is_pad"]][:, :9]
            action_gripper_width = item_dict["action"][~item_dict["is_pad"]][:, 9:10]
            proprio_pose = item_dict["proprio_state"][:9]
            state_gripper_width = item_dict["proprio_state"][9:10] # state

            pose_min = safe_minimum(
                safe_minimum(pose_min, pose), safe_min(action_pose, axis=0)
            )
            pose_max = safe_maximum(
                safe_maximum(pose_max, pose), safe_max(action_pose, axis=0)
            )
            gripper_width_min = safe_minimum(
                safe_minimum(gripper_width_min, state_gripper_width),
                safe_min(action_gripper_width, axis=0),
            )
            gripper_width_max = safe_maximum(
                safe_maximum(gripper_width_max, state_gripper_width),
                safe_max(action_gripper_width, axis=0),
            )
            proprio_min = safe_minimum(
                proprio_min, proprio_pose
            )
            proprio_max = safe_maximum(
                proprio_max, proprio_pose
            )

        obs_normalize_params = {}
        obs_normalize_params["pose"] = {
            "mean": (pose_min + pose_max) / 2,
            "scale": np.maximum((pose_max - pose_min) / 2, scale_eps),
        }
        obs_normalize_params["gripper_width"] = {
            "mean": (gripper_width_min + gripper_width_max) / 2,
            "scale": np.maximum((gripper_width_max - gripper_width_min) / 2, scale_eps),
        }
        obs_normalize_params["proprio_state"] = {
            "mean": (proprio_min + proprio_max) / 2,
            "scale": np.maximum((proprio_max - proprio_min) / 2, scale_eps),
        }
        return obs_normalize_params

    def update_obs_normalize_params(self, obs_normalize_params):
        self.OBS_NORMALIZE_PARAMS = copy.deepcopy(obs_normalize_params)
        pickle.dump(
            obs_normalize_params,
            open(
                os.path.join(
                    self.data_roots[0], f"norm_stats_{len(self.data_roots)}.pkl"
                ),
                "wb",
            ),
        )

        self.pose_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["pose", "gripper_width"]
            ]
        )
        self.pose_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["pose", "gripper_width"]
            ]
        )

        self.proprio_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["proprio_state", "gripper_width"]
            ]
        )
        self.proprio_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["proprio_state", "gripper_width"]
            ]
        )


def step_collate_fn(samples):
    batch = {}
    for key in samples[0].keys():
        if key != "lang":
            # print(key, samples[0][key].shape)
            batched_array = torch.stack([sample[key] for sample in samples], dim=0)
            batch[key] = batched_array
    batch["lang"] = [sample["lang"] for sample in samples]
    return batch


def load_sim2sim_data(data_roots, num_seeds, train_batch_size, val_batch_size, chunk_size, **kwargs):
    # construct dataset and dataloader
    train_dataset = Sim2SimEpisodeDatasetEff(
        data_roots,
        num_seeds,
        split="train",
        chunk_size=chunk_size,
        # norm_stats_path=os.path.join(data_roots[0], f"norm_stats_{len(data_roots)}.pkl"), 
        norm_stats_path=None, 
        **kwargs,
    )
    val_dataset = Sim2SimEpisodeDatasetEff(
        data_roots,
        num_seeds,
        split="val",
        chunk_size=chunk_size,
        norm_stats_path=os.path.join(
            data_roots[0], f"norm_stats_{len(data_roots)}.pkl"
        ),
        **kwargs,
    )
    train_num_workers = 16 #8 
    val_num_workers = 16 #8 
    print(
        f"Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        collate_fn=step_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True,
        collate_fn=step_collate_fn,
    )
    return (
        train_dataloader,
        val_dataloader,
        train_dataset.OBS_NORMALIZE_PARAMS,
        val_dataset.OBS_NORMALIZE_PARAMS,
    )


if __name__ == "__main__":
    print("Testing timing functionality...")
    data_roots =["/home/zhouzhiting/Data/panda_data/cano_policy_pd_3"]
    
    try:
        train_loader, _, _, _ = load_sim2sim_data(
            data_roots=data_roots,
            num_seeds= 10,
            train_batch_size=128,
            val_batch_size=128,
            chunk_size=20,
            usages = ["obs"],
            camera_names=["third"]  #, "wrist"
        )

        for i, batch in enumerate(train_loader):
            print(f"Processed batch {i}")
            if i >= 2:  
                break
                
        print_timing_stats()
        
    except Exception as e:
        print(f"Error testing timing: {e}")
        import traceback
        traceback.print_exc()
