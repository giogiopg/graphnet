"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR

class ORCA115(Detector):
    """`Detector` class for ORCA-115."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._identity, #self._dom_time,
            "pos_x": self._identity, #self._dom_xy,
            "pos_y": self._identity, #self._dom_xy,
            "pos_z": self._identity, #self._dom_z,
            "dir_x": self._identity, #self._dir_xy,
            "dir_y": self._identity, #self._dir_xy,
            "dir_z": self._identity, #self._dir_z,
            "tot": self._identity, #self._tot,
        }
        return feature_map

    def _dom_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10.0

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x - 117.5) / 7.75

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1800) / 180

    def _tot(self, x: torch.tensor) -> torch.tensor:
        # return torch.log10(x)
        return (x - 75) / 7.5

    def _dir_xy(self, x: torch.tensor) -> torch.tensor:
        return x * 10.0

    def _dir_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 0.275) * 12.9


class ORCA6(Detector):
    """`Detector` class for ORCA-6."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._dom_time,
            "pos_x": self._dom_x,
            "pos_y": self._dom_y,
            "pos_z": self._dom_z,
            "dir_x": self._dir_xy,
            "dir_y": self._dir_xy,
            "dir_z": self._dir_z,
            "tot": self._tot,
        }
        return feature_map

    def _dom_x(self, x: torch.tensor) -> torch.tensor:
        return (x - 457.8) * 0.37

    def _dom_y(self, x: torch.tensor) -> torch.tensor:
        return (x - 574.1) * 1.04

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x - 108.6) * 0.12

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1025) * 0.021

    def _tot(self, x: torch.tensor) -> torch.tensor:
        # return torch.log10(x)
        return (x - 117) * 0.085

    def _dir_xy(self, x: torch.tensor) -> torch.tensor:
        return x * 10.0

    def _dir_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 0.23) * 12.9

class ORCA6_2_ORCA115(Detector):
    """`Detector class for ORCA-6 in ORCA115 coordinates."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._identity,
            "pos_x": self._shift_x_pos,
            "pos_y": self._shift_y_pos,
            "pos_z": self._shift_z_pos,
            "dir_x": self._identity,
            "dir_y": self._identity, 
            "dir_z": self._identity, 
            "tot": self._identity,
            }
        return feature_map

    def _shift_x_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA6_x_center = 458.41666159134473
        ORCA115_6_x_center = -50.8455
        return x - (ORCA6_x_center - ORCA115_6_x_center)

    def _shift_y_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA6_y_center = 574.716668752937
        ORCA115_6_y_center = 94.01083333333335
        return x - (ORCA6_y_center - ORCA115_6_y_center)

    def _shift_z_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA6_z_center = 113.86196451110592
        ORCA115_6_z_center = 117.1999120916717
        return x - (ORCA6_z_center - ORCA115_6_z_center)

class ORCA10_2_ORCA115(Detector):
    """`Detector class for ORCA-10 in ORCA115 coordinates."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._identity,
            "pos_x": self._shift_x_pos,
            "pos_y": self._shift_y_pos,
            "pos_z": self._shift_z_pos,
            "dir_x": self._identity,
            "dir_y": self._identity,
            "dir_z": self._identity,
            "tot": self._identity,
            }
        return feature_map

    def _shift_x_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA10_x_center = 461.57000323
        ORCA115_10_x_center = -47.878099999999996
        return x - (ORCA10_x_center - ORCA115_10_x_center)

    def _shift_y_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA10_y_center = 565.32
        ORCA115_10_y_center = 85.05930000000002
        return x - (ORCA10_y_center - ORCA115_10_y_center)

    def _shift_z_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA10_z_center = 108.53536254
        ORCA115_10_z_center = 117.19991209167169
        return x - (ORCA10_z_center - ORCA115_10_z_center)


class ORCA6_v9(Detector):
    """`Detector` class for ORCA-6_v9."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._identity, #self._dom_time,
            "pos_x": self._identity, #self._dom_xy,
            "pos_y": self._identity, #self._dom_xy,
            "pos_z": self._identity, #self._dom_z,
            "dir_x": self._identity, #self._dir_xy,
            "dir_y": self._identity, #self._dir_xy,
            "dir_z": self._identity, #self._dir_z,
            "tot": self._identity, #self._tot,
        }
        return feature_map

