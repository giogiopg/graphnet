"""Code to extract the pulse map information from the KM3NeT ROOT file."""

from typing import Any, Dict
import numpy as np
import pandas as pd

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id_filetype,
    assert_no_uint_values,
    creating_time_zero,
)


class KM3NeTROOTPulseExtractor(KM3NeTROOTExtractor):
    """Class for extracting the entire pulse information from a file."""

    def __init__(self, name: str = "pulse_map"):
        """Initialize the class to extract the pulse information."""
        super().__init__(name)

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract pulse map information and return a dataframe.

        Args:
            file (Any): The file from which to extract pulse map information.

        Returns:
            Dict[str, Any]: A dictionary containing pulse map information.
        """
        pulsemap_df = self._extract_pulse_map(file)
        pulsemap_df = assert_no_uint_values(pulsemap_df)

        return pulsemap_df

    def _extract_pulse_map(self, file: Any) -> pd.DataFrame:
        """Extract the pulse information and assigns unique IDs.

        Args:
            file (Any): The file from which to extract pulse information.

        Returns:
            pd.DataFrame: A dataframe containing pulse information.
        """
        #if data

        if len(file.mc_trks.E[0]>0):
            #evts are neutrinos or muons
            primaries = file.mc_trks[:, 0]
            unique_id = create_unique_id_filetype(
                    np.array(primaries.pdgid),
                    np.array(primaries.E),
                    np.ones(len(primaries.pdgid)),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
            )  # extract the unique_id

        else:
            #evts are noise or data
            if file.header['calibration']=="dynamical":
                #this is data
                unique_id = create_unique_id_filetype(
                    26 * np.ones(len(file.run_id)),
                    np.ones(len(file.run_id)),
                    np.ones(len(file.run_id)),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
            )  # extract the unique_id
            
            else:
                #this is noise
                unique_id = create_unique_id_filetype(
                    np.zeros(len(file.run_id)),
                    np.ones(len(file.run_id)),
                    np.ones(len(file.run_id)),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
            )

       

        hits = file.hits
        keys_to_extract = [
            "t",
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
            "tot",
            "trig",
        ]

        pandas_df = hits.arrays(keys_to_extract, library="pd")
        df = pandas_df.reset_index()
        unique_extended = []
        for index in df["entry"].values:
            unique_extended.append(int(unique_id[index]))
        df["event_no"] = unique_extended
        df = df.drop(["entry", "subentry"], axis=1)
        df = creating_time_zero(df)

        return df
