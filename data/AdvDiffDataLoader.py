import os
import pickle

import numpy as np
import pandas as pd

from .DataLoader import DataLoader


class AdvDiffDataLoader(DataLoader):
    variables = ["temperature", "oxygen", "salinity"]

    def __init__(self, folder_path):
        # By defining __init__ and not calling super().__init__(),
        # the parent's initialization logic is bypassed.
        self.folder_path = folder_path
        self.load_data()
        self.bathy = np.zeros_like(self.height)  # Placeholder for bathymetry if needed

    def load_data(self):
        data = {}
        for var in self.variables:
            file_path = os.path.join(self.folder_path, f"{var}.pkl")
            with open(file_path, "rb") as f:
                data[var] = pickle.load(f)
        df = pd.read_csv(os.path.join(self.folder_path, "data.csv"))[
            ["Longitude", "Latitude"]
        ]
        df = df[
            (df["Longitude"] != 0) & (df["Latitude"] != 0)
        ]  # Filter out rows with 0 values

        path = np.asarray(df)
        self.data = data
        self.set_fields()
        self.data["vehicle_path"], (self.X_coord, self.Y_coord) = (
            self.offset_path_meshgrid_with_meshgrid(
                path, self.data["temperature"]["meshgrid"]
            )
        )

        return self.data

    def set_fields(self):
        self.UU = self.data["temperature"]["solutions"]
        self.VV = self.data["oxygen"]["solutions"]
        self.height = self.data["salinity"]["solutions"]
        # self.X_coord, self.Y_coord = self.data["temperature"]["meshgrid"]
        self.solutions = np.stack([self.height, self.UU, self.VV], axis=-1).transpose(
            0, 3, 2, 1
        )[:1000]
        # self.solutions = self.solutions / np.max(
        #     np.abs(self.solutions)
        # )  # Normalize for better numerical stability
        print(f"solutions shape: {self.solutions.shape}")

    def offset_path_meshgrid_with_meshgrid(self, path, meshgrid):
        # Calculate center origin for the transformation

        lon0 = np.min(meshgrid[0])
        lat0 = np.min(meshgrid[1])
        # Conversion factor:    1 degree approx 111111 meters
        self.coord_factor = (111111, 111111)
        self.coord_origin = (
            lon0,
            lat0,
        )
        # self.coord_origin = (lon0, lat0)
        # Scale and offset path (assuming path[:, 0] is Lat, path[:, 1] is Lon)
        # to match meshgrid meters space
        x_meters = (path[:, 0] - lon0) * self.coord_factor[0]
        y_meters = (path[:, 1] - lat0) * self.coord_factor[1]
        x, y = meshgrid
        x = (x - lon0) * self.coord_factor[0]
        y = (y - lat0) * self.coord_factor[1]

        return np.stack([x_meters, y_meters], axis=1), (x, y)


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    folder_path = "data/bbc_pier"
    loader = AdvDiffDataLoader(folder_path)
    data = loader.load_data()
    print(data["temperature"].keys())
    print(data["temperature"]["mesh_mask"][40])
