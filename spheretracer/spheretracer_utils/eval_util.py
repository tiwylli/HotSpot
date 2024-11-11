import torch
import numpy  as np

def generate_eval_cams(img_size, radius, num_views, height, fov, batch_size):
    poses = generate_camera_poses(radius=radius, num_views=num_views, height=height)
    uv_batches = create_uv(width=img_size[0], height=img_size[1], batch_size=batch_size)
    intrinsics = create_camera_intrinsics(fov=fov, width=img_size[0], height=img_size[1])
    return poses, uv_batches, intrinsics

def create_camera_intrinsics(fov, width, height, device='cuda:0'):
    aspect_ratio = width / height
    fov_rad = np.radians(fov)
    if width > height:
        f_y = (height / 2) / torch.tan(torch.tensor(fov_rad/2, device=device))
        f_x = f_y * aspect_ratio
    else:
        f_x = (width / 2) / torch.tan(torch.tensor(fov_rad/2, device=device))
        f_y = f_x * aspect_ratio
    intrinsics = torch.tensor([[f_x, 0, width/2, 0],
                               [0, f_y, height/2, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)
    return intrinsics

def generate_camera_poses(radius, num_views, height, device='cuda:0'):
    poses = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        position = torch.tensor([x, height, z], device=device, dtype=torch.float32)

        # Calculate direction vectors
        forward = -position / torch.norm(position)
        up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
        right = torch.cross(up, forward)
        right = right / torch.norm(right)
        up = -torch.cross(forward, right)

        # Create the 4x4 transformation matrix
        pose_matrix = torch.eye(4, device=device)
        pose_matrix[:3, 0] = right
        pose_matrix[:3, 1] = up
        pose_matrix[:3, 2] = forward
        pose_matrix[:3, 3] = position
        poses.append(pose_matrix.unsqueeze(0))
    poses = torch.stack(poses)
    return poses

def create_uv(width, height, batch_size=10000, device='cuda:0'):
    u = torch.arange(width, device=device)
    v = torch.arange(height, device=device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u_grid, v_grid], dim=-1).float().view(-1, 2)
    uv = uv.unsqueeze(0) 
    uv = uv.view(-1, 2)
    uv_batches = [uv[i:i + batch_size].unsqueeze(0) for i in range(0, uv.shape[0], batch_size)]
    return uv_batches