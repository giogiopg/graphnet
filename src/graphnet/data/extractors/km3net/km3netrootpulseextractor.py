"""Code to extract the pulse map information from the KM3NeT ROOT file."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id,
    assert_no_uint_values,
    creating_time_zero,
    creating_time_zero_noise,
    remove_duplicated_event_no,
    noise_selection,
)


class KM3NeTROOTPulseExtractor(KM3NeTROOTExtractor):
    """Class for extracting the entire pulse information from a file."""

    def __init__(
                    self, 
                    name: str = "pulse_map", 
                    DOMs_dict: Optional[dict] = None, 
                    time_window: Optional[Tuple[float, float]] = (-1000.0, 1000.0), 
                    max_noise: Tuple[int, int] = (150, 150)
    ):
        """Initialize the class to extract the pulse information."""
        super().__init__(name)
        self.DOMs_dict = DOMs_dict
        self.time_window = time_window
        self.max_noise = max_noise

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
        ) 

        unique_id_df = pd.DataFrame()
        unique_id_df['event_no'] = unique_id
        unique_id_df = remove_duplicated_event_no(unique_id_df, col = 'event_no', keep = 'first')
        unique_id = unique_id_df.values

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
                                "dom_id",
                                "channel_id"
        ]

        pandas_df = hits.arrays(keys_to_extract, library="pd")
        
        df = pandas_df.reset_index()
        unique_extended = []
        for index in df["entry"].values:
            unique_extended.append(int(unique_id[index]))
           
        df["event_no"] = unique_extended
        df = df.drop(["entry", "subentry"], axis=1)
        
        if self.time_window or self.max_noise is not None:
            df = noise_selection(df, self.time_window, self.max_noise)
        
        if self.DOMs_dict is not None:
            df["du_id"] = list(map(self.DOMs_dict.get, df["dom_id"].astype(str)))

        df = creating_time_zero_noise(df)

        return df