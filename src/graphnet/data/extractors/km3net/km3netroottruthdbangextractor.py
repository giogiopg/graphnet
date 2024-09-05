"""Code to extract the truth event information from the KM3NeT ROOT file."""

from typing import Any, Dict
import numpy as np
import pandas as pd
import km3io as ki

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id_dbang,
    xyz_dir_to_zen_az,
    assert_no_uint_values,
)


class KM3NeTROOTTruthDBangExtractor(KM3NeTROOTExtractor):
    """Class for extracting the truth information from a file."""

    def __init__(self, name: str = "truth"):
        """Initialize the class to extract the truth information."""
        super().__init__(name)

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract truth event information as a dataframe."""
        truth_df = self._extract_truth_dataframe(file)
        truth_df = assert_no_uint_values(truth_df)  # asserts the data format

        return truth_df

    def _extract_truth_dataframe(self, file: Any) -> Any:
        """Extract truth information from a file and returns a dataframe.

        Args:
            file (Any): The file from which to extract truth information.

        Returns:
            pd.DataFrame: A dataframe containing truth information.
        """
        #check if the pos_x 0 entry has two values or just one
        padding_value = 999.0
        if (len(file.mc_trks.pdgid[0]) == 2) and (file.mc_trks.pdgid[0][0] == file.mc_trks.pdgid[0][1]) and (file.mc_trks.pdgid[0][0] in [11, 111, 211]):
            #double cascade
            is_double_cascade = np.ones(len(file.mc_trks.pos_x), dtype=int)
            #compute the distance between the two cascades
            distance = np.sqrt(
                (file.mc_trks.pos_x[:, 0] - file.mc_trks.pos_x[:, 1]) ** 2
                + (file.mc_trks.pos_y[:, 0] - file.mc_trks.pos_y[:, 1]) ** 2
                + (file.mc_trks.pos_z[:, 0] - file.mc_trks.pos_z[:, 1]) ** 2
            )
        else:
            #single cascade
            is_double_cascade = np.zeros(len(file.mc_trks.pos_x), dtype=int)
            distance = np.zeros(len(file.mc_trks.pos_x), dtype=float)

        

        primaries = file.mc_trks[:, 0]

        zen_truth, az_truth = xyz_dir_to_zen_az(
            np.array(primaries.dir_x),
            np.array(primaries.dir_y),
            np.array(primaries.dir_z),
            padding_value,
        )
        part_dir_x, part_dir_y, part_dir_z = (
            np.array(primaries.dir_x),
            np.array(primaries.dir_y),
            np.array(primaries.dir_z),
        )
        unique_id = create_unique_id_dbang(
            np.array(primaries.E),  
            np.array(primaries.pos_x),  
            np.array(file.id)
        ) 

        dict_truth = {
            "pdgid": np.array(primaries.pdgid),
            "vrx_x": np.array(primaries.pos_x),
            "vrx_y": np.array(primaries.pos_y),
            "vrx_z": np.array(primaries.pos_z),
            "zenith": zen_truth,
            "azimuth": az_truth,
            "part_dir_x": part_dir_x,
            "part_dir_y": part_dir_y,
            "part_dir_z": part_dir_z,
            "Energy": np.array(primaries.E),
            "n_hits": np.array(file.n_hits),
            "is_double_cascade": is_double_cascade,
            "distance": np.array(distance),
            "event_no": np.array(unique_id).astype(int),
        }

        truth_df = pd.DataFrame(dict_truth)

        return truth_df
