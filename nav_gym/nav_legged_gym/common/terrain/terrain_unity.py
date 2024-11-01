import os
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix
from isaacgym import gymapi

from nav_gym.nav_legged_gym.utils.warp_utils import convert_to_wp_mesh
import torch

from nav_gym import NAV_GYM_ROOT_DIR
class TerrainUnity:
    def __init__(self,gym,sim,device,num_envs,env_spacing=2.0):

        self.gym = gym
        self.sim = sim
        self.device = device
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        # Load your terrain mesh here
        asset_root = os.path.join(NAV_GYM_ROOT_DIR,"resources")
        terrain_file = "/terrain/CombinedMesh_v4_3.obj"
        if(os.path.exists(asset_root + terrain_file)):
            print("[INFO]Terrain file found")
        else:
            print("[ERROR]Terrain file not found")
            exit(1)

        # Load the terrain mesh from a .obj file
        self.terrain_mesh = trimesh.load(asset_root + terrain_file)  # Replace 'terrain.obj' with your mesh file path

        # Calculate the center of the mesh
        mesh_center = self.terrain_mesh.centroid
        x=-10.0#-98.0
        y=10.0#2.0
        z=0.0
        translation = np.array([-y, z, x])# standard to unity vector conversion: x_u,y_u,z_u->-y,z,x

        # Optionally, translate the mesh so that its center is at the origin
        # self.terrain_mesh.apply_translation(-mesh_center + translation)
        print(f"[INFO]Terrain Center: {mesh_center}")
        print(f"[INFO]Terrain Translation: {translation}")
        self.terrain_mesh.apply_translation(translation)

        # Set the heading angle in degrees
        heading_angle_degrees = 90  # Replace with your desired angle in degrees
        heading_angle_radians = np.deg2rad(heading_angle_degrees)

        # Set the rotation axis (e.g., Z-axis)
        rotation_axis = [1.0, 0.0, 0.0]  # Rotate around Z-axis

        # Create the rotation matrix
        print("[INFO]Rotating Terrain:heading_angle_degrees{heading_angle_degrees} rotation_axis{rotation_axis}")
        R = rotation_matrix(heading_angle_radians, rotation_axis)

        # Apply the rotation to the mesh
        self.terrain_mesh.apply_transform(R)

        # Extract vertices and triangle indices from the mesh
        self.vertices = np.array(self.terrain_mesh.vertices, dtype=np.float32)
        self.triangles = np.array(self.terrain_mesh.faces, dtype=np.uint32)

        # Convert to wp mesh
        self.wp_meshes = convert_to_wp_mesh(self.terrain_mesh.vertices, self.terrain_mesh.faces, self.device)

        # Create triangle mesh parameters
        self.tm_params = gymapi.TriangleMeshParams()
        self.tm_params.nb_vertices = self.vertices.shape[0]
        self.tm_params.nb_triangles = self.triangles.shape[0]

        # Initialize the transform without any rotation
        self.tm_params.transform = gymapi.Transform()
        self.tm_params.transform.p = gymapi.Vec3(0.0, 0.0, 0.0)  # dON'T CHANGE THIS, WILL CAUSE DIFFERNECE IN WP MESH

        # Set friction and restitution
        self.tm_params.static_friction = 0.5
        self.tm_params.dynamic_friction = 0.5
        self.tm_params.restitution = 0.0

        # Get the environment origins
        # self._calcu_env_origins_grid()
        self._calcu_env_origins_custom()

    def add_to_sim(self):
        # Add the terrain mesh to the simulation
        self.gym.add_triangle_mesh(self.sim, self.vertices.flatten(), self.triangles.flatten(), self.tm_params)

    def _calcu_env_origins_custom(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)  # Dim:(num_envs, 3)
        # set origin and goal positions
        self.x_goal = 10.0
        self.y_goal =-40.0
        self.x_origin=0.0
        self.y_origin=0.0
        self.env_origins[:, 0] = torch.tensor([self.x_origin] * self.num_envs, device=self.device)
        self.env_origins[:, 1] = torch.tensor([self.y_origin] * self.num_envs, device=self.device)
        # Remove the hardcoded z value
        # self.env_origins[:, 2] = -1.0

        # Prepare ray origins and directions
        z_max = 60.0  # Set a height above the highest possible terrain point
        ray_origins = np.zeros((self.num_envs, 3))
        ray_origins[:, 0] = self.env_origins[:, 0].cpu().numpy()
        ray_origins[:, 1] = self.env_origins[:, 1].cpu().numpy()
        ray_origins[:, 2] = z_max  # Start raycasting from above

        ray_directions = np.zeros((self.num_envs, 3))
        ray_directions[:, 2] = -1.0  # Pointing downwards along the z-axis

        # Perform raycasting
        ray_intersections, index_ray, index_tri = self.terrain_mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )

        num_rays = ray_origins.shape[0]
        highest_intersections = np.zeros((num_rays, 3))

        # Loop through each ray to find the intersection with the largest z-value
        for ray_idx in range(num_rays):
            # Get the indices of intersections for the current ray
            indices = np.where(index_ray == ray_idx)[0]
            
            if len(indices) == 0:
                # Handle rays that did not intersect the mesh
                print(f"[WARNING] Ray {ray_idx} did not hit the terrain")
                # You can set a default z-value or handle it as needed
                highest_intersections[ray_idx] = [ray_origins[ray_idx, 0], ray_origins[ray_idx, 1], 0.0]  # Default z=0.0
                continue
            
            # Get all intersections for this ray
            intersections = ray_intersections[indices]
            
            # Find the intersection with the maximum z-value
            max_z_idx = np.argmax(intersections[:, 2])
            highest_intersections[ray_idx] = intersections[max_z_idx]

        # Update env_origins z-coordinate with the terrain height
        self.env_origins[:, 2] = torch.tensor(highest_intersections[:, 2], device=self.device)

    def _calcu_env_origins_grid(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)  # Dim:(num_envs, 3)
        # Create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        # Remove the hardcoded z value
        # self.env_origins[:, 2] = -1.0

        # Prepare ray origins and directions
        z_max = 60.0  # Set a height above the highest possible terrain point
        ray_origins = np.zeros((self.num_envs, 3))
        ray_origins[:, 0] = self.env_origins[:, 0].cpu().numpy()
        ray_origins[:, 1] = self.env_origins[:, 1].cpu().numpy()
        ray_origins[:, 2] = z_max  # Start raycasting from above

        ray_directions = np.zeros((self.num_envs, 3))
        ray_directions[:, 2] = -1.0  # Pointing downwards along the z-axis

        # Perform raycasting
        ray_intersections, index_ray, index_tri = self.terrain_mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )

        num_rays = ray_origins.shape[0]
        highest_intersections = np.zeros((num_rays, 3))

        # Loop through each ray to find the intersection with the largest z-value
        for ray_idx in range(num_rays):
            # Get the indices of intersections for the current ray
            indices = np.where(index_ray == ray_idx)[0]
            
            if len(indices) == 0:
                # Handle rays that did not intersect the mesh
                print(f"[WARNING] Ray {ray_idx} did not hit the terrain")
                # You can set a default z-value or handle it as needed
                highest_intersections[ray_idx] = [ray_origins[ray_idx, 0], ray_origins[ray_idx, 1], 0.0]  # Default z=0.0
                continue
            
            # Get all intersections for this ray
            intersections = ray_intersections[indices]
            
            # Find the intersection with the maximum z-value
            max_z_idx = np.argmax(intersections[:, 2])
            highest_intersections[ray_idx] = intersections[max_z_idx]

            #offset
            z_offset = torch.tensor([0.2]).repeat(self.env_origins.shape[0]).to(self.device)

        # Update env_origins z-coordinate with the terrain height
        self.env_origins[:, 2] = torch.tensor(highest_intersections[:, 2], device=self.device) + z_offset

    def sample_new_init_poses(self,env_ids):
        # Sample new initial poses for the environments
        return self.env_origins[env_ids] 

