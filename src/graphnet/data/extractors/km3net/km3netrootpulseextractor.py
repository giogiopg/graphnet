"""Code to extract the pulse map information from the KM3NeT ROOT file."""

from typing import Any, Dict
import numpy as np
import pandas as pd
import awkward as ak

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id,
    mask_saturated_pmts,
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
        unique_id = create_unique_id(
            np.array(file.run_id),
            np.array(file.frame_index),
            np.array(file.trigger_counter),
        )  # creates the unique_id

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
        mask_saturated_pmts = False
        if mask_saturated_pmts:
            keys_to_extract.append("channel_id")
            keys_to_extract.append("dom_id")

        pandas_df = ak.to_dataframe(hits.arrays(keys_to_extract, library="ak"))
        df = pandas_df.reset_index()
        unique_extended = []
        for index in df["entry"].values:
        #for index in df["index"].values:
            unique_extended.append(int(unique_id[index]))
        df["event_no"] = unique_extended
        # keep only non saturated pmts or DOMs
        if mask_saturated_pmts:
            df = mask_saturated_pmts(df)
            df = df.drop(["channel_id", "dom_id"], axis=1)

        df = df.drop(["entry", "subentry"], axis=1)
        df = creating_time_zero(df)
        print(df)

        return df
