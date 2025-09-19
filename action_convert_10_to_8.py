import numpy as np

from transforms3d.euler import mat2euler




#action is 8 dim without Batch

# action = np.array([0,0,0,1,0,0,0,1])

action = np.array([[0,0,0,1,0,0,1,0,0,1],
                  [0,0,0,1,0,0,1,0,0,1]])


def mat_6_to_mat(mat_6):
    assert mat_6.shape[-1] == 2
    assert mat_6.shape[-2] == 3
    
    mat_6[:, :, 0] = mat_6[:, :, 0] / np.linalg.norm(mat_6[:, :, 0]) # [B, 3]
    mat_6[:, :, 1] = mat_6[:, :, 1] / np.linalg.norm(mat_6[:, :, 1]) # [B, 3]
    z_vec = np.cross(mat_6[:, :, 0], mat_6[:, :, 1]) # [B, 3]
    z_vec = z_vec[:, :, np.newaxis]  # (B, 3, 1)
    mat = np.concatenate([mat_6, z_vec], axis=2) # [B, 3, 3]
    
    return mat

    



action_pos, action_mat_6, gripper_width = action[:,:3], action[:,3:9], action[:,9:]

mat = mat_6_to_mat(action_mat_6.reshape(-1,3,2))

action_euler = []
for i in range(mat.shape[0]):
    action_euler.append(mat2euler(mat[i]))
    

action_euler = np.stack(action_euler)

action_8d = np.concatenate([action_pos, action_euler, gripper_width], axis=-1)

print("action_8d:", action_8d)
