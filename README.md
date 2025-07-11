# Diffusion-policy

- Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
- collect data by homebot/homebotsapien/collect_il_real.py(https://github.com/zhouzt21/homebot/tree/main)

## Network 

- scripts/policy.py `We use`
    - backbones: ResNet18Conv
    - pools: SpatialSoftmax
    - linears
    - noise_pred_net: ConditionalUnet1D

## Datasets

- Sim2SimEpisodeDataset (dataset.py)
    - 每个step的结构数据分开存储，图片每步存储单张.jpg
- Sim2SimEpisodeDatasetEff (dataset_array.py) `We use`
    - 是将所有step数据（不包括图片）全部存储成一个total_steps.npz的实现，图片仍每步存储单张.jpg，可加速读取，目前使用

### result_dict 

The output of `get_unnormalized_item()`
- result_dict["action"] 
    - 这里是已经做了chunk处理的action_chunk -> shape (chuck_size, 10(3(xyz)+6(rotation)+1(gripper)))
    - use_desired_action: 则是用action相对上一目标pose计算出的作为target pose; 如果不用，则是用tcp_pose作为target pose (maybe false)
    - 最后都是相对tcp_pose_at_obs来计算的delta_pose (maybe false)
- result_dict["proprio_state"]  
    - proprio_state, 一直都是观察帧的tcp pose，重复chunk个用来对齐 (maybe false)
    - ndarray, (chunk_size, 10d)
- result_dict["image"]
    - [cam_num, height(width), width(height), 3]


## normalization

- chunk处理之后的norm操作
    - [proprio_state]
        - data["tcp_pose"] + gripper_width
    - [pose]
        - action 的最小值 和 robot_state的最小值中的最小值. Note: `robot state` not use  

- Note: unnormalize in evaluation (homebot)
    - ref: [git@github.com:pancake-w/ManiSkill_evaluation.git](https://github.com/pancake-w/ManiSkill_evaluation/tree/dev-bingwen) (in dev-bingwen branch) mani_skill/evaluation/policies/diffusion_policy/dp_infer.py
    - 仿真得到的pose_at_obs是7d，转10d输入，没有归一化 (maybe false)
    - 得到的action 10d -> 4*4d，然后和输入时的pose_at_obs转的4 *4d计算后，得到desired delta pose(7d+2d空)作为step需要的，没有归一化 (maybe false)