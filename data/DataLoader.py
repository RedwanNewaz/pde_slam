import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# solutions[:, 0, ...]
class DataLoader:
    def __init__(self, source):
        self.source = source
        self.bathy = np.load(os.path.join(self.source, "bathymetry_values.npy"))
        self.solutions = np.load(os.path.join(self.source, "solutions.npy"))
        self.X_coord = np.load(os.path.join(self.source, "X_coords.npy"))
        self.Y_coord = np.load(os.path.join(self.source, "Y_coords.npy"))

        self.height, self.UU, self.VV = (
            self.solutions[:, 0, ...],
            self.solutions[:, 1, ...],
            self.solutions[:, 2, ...],
        )

    def getFreeSurfaceAt(self, t):
        N = self.height.shape[0]
        h = self.height[t % N]
        # free_surface = h + self.bathy
        # free_surface[h < 1.0e-3] = np.nan
        # h[h < 1.0e-3] = np.nan
        free_surface = h
        return free_surface

    def __len__(self):
        return self.X_coord.shape[0] * self.Y_coord.shape[1]

    def getUAt(self, t):
        N = self.UU.shape[0]
        return self.UU[t % N]

    def getVAt(self, t):
        N = self.VV.shape[0]
        return self.VV[t % N]

    def extendDomain(self, factor):
        times = np.arange(self.solutions.shape[0])
        x_coords, y_coords = (
            np.arange(
                self.X_coord.shape[0],
            ),
            np.arange(self.Y_coord.shape[1]),
        )
        x_new_coords = np.linspace(
            x_coords[0], x_coords[-1], int(len(x_coords) * factor)
        )
        y_new_coords = np.linspace(
            y_coords[0], y_coords[-1], int(len(y_coords) * factor)
        )
        if hasattr(self, "data"):
            print(np.max(self.X_coord))
            path_factor_x = np.max(self.X_coord) / self.X_coord.shape[0]
            path_factor_y = np.max(self.Y_coord) / self.Y_coord.shape[0]
            self.data["vehicle_path"][:, 0] /= path_factor_x
            self.data["vehicle_path"][:, 1] /= path_factor_y
            if hasattr(self, "coord_factor"):
                self.coord_factor = (
                    self.coord_factor[0] / path_factor_x,
                    self.coord_factor[1] / path_factor_y,
                )

            # self.data["vehicle_path"] = self.data["vehicle_path"][:, [1, 0]]
            self.mesh_mask = self.data["temperature"]["mesh_mask"].T
        # else:
        #     self.coord_factor = 111111
        #     self.coord_origin = (np.min(self.X_coord), np.min(self.Y_coord))

        new_solutions = np.zeros(
            (
                self.solutions.shape[0],
                self.solutions.shape[1],
                len(x_new_coords),
                len(y_new_coords),
            )
        )
        for i in range(3):
            interpolator = RegularGridInterpolator(
                (times, x_coords, y_coords),
                self.solutions[:, i, ...].transpose(0, 2, 1),
                bounds_error=False,
                fill_value=0,
                method="linear",
            )
            tt, x_mesh, y_mesh = np.meshgrid(times, x_coords, y_coords, indexing="ij")

            # assert np.allclose(
            #     self.solutions[:, i, ...], interpolator((tt, y_mesh, x_mesh))
            # )
            tt, x_mesh, y_mesh = np.meshgrid(
                times, x_new_coords, y_new_coords, indexing="ij"
            )
            new_solutions[:, i, ...] = interpolator((tt, y_mesh, x_mesh))
        self.X_coord, self.Y_coord = np.meshgrid(
            x_new_coords, y_new_coords, indexing="ij"
        )
        self.solutions = new_solutions
        self.height, self.UU, self.VV = (
            self.solutions[:, 0, ...],
            self.solutions[:, 1, ...],
            self.solutions[:, 2, ...],
        )

    def size(self):
        return self.solutions.shape

    def __str__(self):
        return f"solution size: {self.size()}\nBathymetry shape: {self.bathy.shape}\n"

    def __repr__(self):
        return self.__str__()
