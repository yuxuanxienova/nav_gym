import math
import numpy as np
import torch
import cv2

from isaacgym import gymapi, gymutil


class BatchWireframeSphereGeometry(gymutil.LineGeometry):
    """Draw multiple spheres without a for loop"""

    def __init__(self, num_spheres, radius=1.0, num_lats=8, num_lons=8, pose=None, color=None, color2=None):
        if color is None:
            color = (1, 0, 0)

        if color2 is None:
            color2 = color

        self.num_lines = 2 * num_lats * num_lons
        self.num_spheres = num_spheres

        verts = np.empty((self.num_lines, 2), gymapi.Vec3.dtype)
        colors = np.empty(self.num_lines, gymapi.Vec3.dtype)
        idx = 0

        ustep = 2 * math.pi / num_lats
        vstep = math.pi / num_lons

        u = 0.0
        for i in range(num_lats):
            v = 0.0
            for j in range(num_lons):
                x1 = radius * math.sin(v) * math.sin(u)
                y1 = radius * math.cos(v)
                z1 = radius * math.sin(v) * math.cos(u)

                x2 = radius * math.sin(v + vstep) * math.sin(u)
                y2 = radius * math.cos(v + vstep)
                z2 = radius * math.sin(v + vstep) * math.cos(u)

                x3 = radius * math.sin(v + vstep) * math.sin(u + ustep)
                y3 = radius * math.cos(v + vstep)
                z3 = radius * math.sin(v + vstep) * math.cos(u + ustep)

                verts[idx][0] = (x1, y1, z1)
                verts[idx][1] = (x2, y2, z2)
                colors[idx] = color

                idx += 1

                verts[idx][0] = (x2, y2, z2)
                verts[idx][1] = (x3, y3, z3)
                colors[idx] = color2

                idx += 1

                v += vstep
            u += ustep

        if pose is None:
            self.verts = np.repeat(verts, num_spheres, axis=0)
        else:
            self.verts = pose.transform_points(verts)

        self.verts_tmp = np.copy(self.verts)
        self._colors = np.repeat(colors, num_spheres, axis=0)

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors

    def draw(self, positions, gym, viewer, env, colors=None):
        if len(positions.shape) == 2:
            positions = positions.unsqueeze(0)
        flat_pos = positions.unsqueeze(0).repeat(self.num_lines, 1, 1, 1).view(-1, 3).cpu().numpy()
        self.verts_tmp["x"][:, 0] = self.verts["x"][:, 0] + flat_pos[:, 0]
        self.verts_tmp["x"][:, 1] = self.verts["x"][:, 1] + flat_pos[:, 0]
        self.verts_tmp["y"][:, 0] = self.verts["y"][:, 0] + flat_pos[:, 1]
        self.verts_tmp["y"][:, 1] = self.verts["y"][:, 1] + flat_pos[:, 1]
        self.verts_tmp["z"][:, 0] = self.verts["z"][:, 0] + flat_pos[:, 2]
        self.verts_tmp["z"][:, 1] = self.verts["z"][:, 1] + flat_pos[:, 2]
        if colors is None:
            colors = self._colors
        else:
            vc = np.empty(self.num_spheres, gymapi.Vec3.dtype)
            for i in range(self.num_spheres):
                vc[i][0] = colors[i, 0]
                vc[i][1] = colors[i, 1]
                vc[i][2] = colors[i, 2]
            vc = np.tile(vc, (1, self.num_lines))
            colors = vc
        gym.add_lines(viewer, env, self.num_spheres * self.num_lines, self.verts_tmp, colors)


def show_depth_img(images, res=120, max_d=4.0):
    """
    Show depth image
    Args:
        images: depth image (num_camera, H, W)
        res: resolution of the visualization image.
        max_d: maximum depth value to show. Used for normalization.
    """
    n_camera = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    img = torch.cat([images[i] for i in range(images.shape[0])], dim=1)
    img = img.clamp(0.0, max_d)
    img = ((img.cpu().numpy() / max_d) * 255).astype("uint8")
    mask = img > 230
    # gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    gray[mask] = (0, 0, 0)

    res_h = res
    res_w = int(res * w / h)
    gray = cv2.resize(gray, dsize=(res_w * n_camera, res_h))
    cv2.imshow("window", gray)
    cv2.waitKey(1)


def destroy_depth_window():
    try:
        if cv2.getWindowProperty("window", cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow("window")
    except:
        pass