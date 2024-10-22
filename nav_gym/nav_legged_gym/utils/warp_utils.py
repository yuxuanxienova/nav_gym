import numpy as np
import torch

import warp as wp
from warp.torch import to_torch

wp.init()


@wp.kernel
def raycast_kernel(
    mesh: wp.uint64,
    ray_starts_world: wp.array(dtype=wp.vec3),
    ray_directions_world: wp.array(dtype=wp.vec3),
    ray_hits_world: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=float),
):

    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index
    max_dist = float(1e6)  # max raycast disance
    # ray cast against the mesh
    if wp.mesh_query_ray(mesh, ray_starts_world[tid], ray_directions_world[tid], max_dist, t, u, v, sign, n, f):
        ray_hits_world[tid] = ray_starts_world[tid] + t * ray_directions_world[tid]
        ray_distance[tid] = t


def ray_cast(ray_starts_world, ray_directions_world, wp_mesh):
    """Performs ray casting on the terrain mesh.

    Args:
        ray_starts_world (Torch.tensor): The starting position of the ray.
        ray_directions_world (Torch.tensor): The ray direction.

    Returns:
        [Torch.tensor]: The ray hit position. Returns float('inf') for missed hits.
        [Torch.tensor]: The ray hit distance. Returns float('inf') for missed hits.

    """
    shape = ray_starts_world.shape
    ray_starts_world = ray_starts_world.view(-1, 3)
    ray_directions_world = ray_directions_world.view(-1, 3)
    num_rays = len(ray_starts_world)
    ray_starts_world_wp = wp.types.array(
        ptr=ray_starts_world.data_ptr(),
        dtype=wp.vec3,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=wp_mesh.device,
    )
    ray_directions_world_wp = wp.types.array(
        ptr=ray_directions_world.data_ptr(),
        dtype=wp.vec3,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=wp_mesh.device,
    )
    ray_distance = torch.zeros((num_rays, 1), device=ray_starts_world.device)
    ray_distance[:] = float("inf")
    ray_distance_wp = wp.types.array(
        ptr=ray_distance.data_ptr(),
        dtype=float,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=wp_mesh.device,
    )
    ray_hits_world = torch.zeros((num_rays, 3), device=ray_starts_world.device)
    ray_hits_world[:] = float("inf")
    ray_hits_world_wp = wp.types.array(
        ptr=ray_hits_world.data_ptr(), dtype=wp.vec3, shape=(num_rays,), copy=False, owner=False, device=wp_mesh.device
    )
    wp.launch(
        kernel=raycast_kernel,
        dim=num_rays,
        inputs=[wp_mesh.id, ray_starts_world_wp, ray_directions_world_wp, ray_hits_world_wp, ray_distance_wp],
        device=wp_mesh.device,
    )
    wp.synchronize()
    return ray_hits_world.view(shape), ray_distance.view(*shape[:-1], 1)


def convert_to_wp_mesh(vertices, triangles, device):
    return wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(triangles.astype(np.int32).flatten(), dtype=int, device=device),
    )