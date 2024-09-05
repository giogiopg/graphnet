"""Code for extracting the montecarlo pulse information from a file. Code for prem studies of the HNL."""

from typing import Any, Dict
import numpy as np
import pandas as pd

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id_dbang,
    assert_no_uint_values,
    creating_time_zero,
    pmt_id_to_pos_dir,
)


class KM3NeTMCPulseExtractor(KM3NeTROOTExtractor):
    """Class for extracting the mc pulse information from a file."""

    def __init__(self, name: str = "mc_pulse_map"):
        """Initialize the class to extract the mc pulse information."""
        super().__init__(name)

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract pulse map information and return a dataframe.

        Args:
            file (Any): The file from which to extract mc pulse map information.

        Returns:
            Dict[str, Any]: A dictionary containing mc pulse map information.
        """
        pulsemap_df = self._extract_pulse_map(file)
        pulsemap_df = assert_no_uint_values(pulsemap_df)

        return pulsemap_df

    def _extract_pulse_map(_: Any, file: Any) -> pd.DataFrame:
        """Extract the pulse information and assigns unique IDs.

        Args:
            file (Any): The file from which to extract pulse information.

        Returns:
            pd.DataFrame: A dataframe containing pulse information.
        """
        primaries = file.mc_trks[:, 0]
        unique_id = create_unique_id_dbang(
            np.array(primaries.E),  
            np.array(primaries.pos_x),  
            np.array(file.id)
        ) 

        mc_hits = file.mc_hits
        keys_to_extract = [
            "t",
            "pmt_id",
        ]

        pandas_df = mc_hits.arrays(keys_to_extract, library="pd")
        df = pandas_df.reset_index()
        unique_extended = []
        for index in df["entry"].values:
            unique_extended.append(int(unique_id[index]))
        df["event_no"] = unique_extended

        df = df.drop(["entry", "subentry"], axis=1)
        df = creating_time_zero(df)

        #extract from the pmt_id the position and direction of the hit
        df = pmt_id_to_pos_dir(df, det_file='/pbs/throng/km3net/detectors/orca_115strings_av20min17mhorizontal_18OMs_alt9mvertical_v2.detx') #manual, TODO: change to a more general way

        return df
