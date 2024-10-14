"""Code for extracting the triggered pulse information from a file."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id,
    create_unique_id_filetype,
    assert_no_uint_values,
    creating_time_zero,
    creating_time_zero_noise,
    remove_duplicated_event_no,
    noise_selection,
)


class KM3NeTROOTTriggPulseExtractorORCA(KM3NeTROOTExtractor):
    """Class for extracting the triggered pulse information from a file."""

    def __init__(self, name: str = "pulse_map", DOMs_dict: Optional[dict] = None):
        """Initialize the class to extract the pulse information."""
        super().__init__(name)
        self.DOMs_dict = DOMs_dict

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract pulse map information and return a dataframe.

        Args:
            file (Any): The file from which to extract triggered pulse map information.

        Returns:
            Dict[str, Any]: A dictionary containing triggered pulse map information.
        """
        pulsemap_df = self._extract_pulse_map(file)
        pulsemap_df = assert_no_uint_values(pulsemap_df)

        return pulsemap_df

    def _extract_pulse_map(self: Any, file: Any) -> pd.DataFrame:
        """Extract the pulse information and assigns unique IDs.

        Args:
            file (Any): The file from which to extract pulse information.

        Returns:
            pd.DataFrame: A dataframe containing pulse information.

        Analogous to the KM3NeTROOTPulseExtractor but doing cuts on trig.
        """
        n_evts = len(file)
        padding_value = 999.0 
        ones_vector = np.ones(n_evts)
        padding_vector = padding_value * ones_vector
        nus_flavor = [12, 14, 16]
        
        if abs(np.array(file.mc_trks[0, 0].pdgid)) not in nus_flavor:
            # Muon file
            primaries = file.mc_trks[:, 1]
            is_cc_flag = padding_vector

        elif abs(np.array(file.mc_trks[0, 0].pdgid)) in nus_flavor:
            # Neutrino file
            primaries = file.mc_trks[:, 0]
            is_cc_flag = np.array(file.w2list[:, 10] == 2)
            
        primaries = file.mc_trks[:, 0]
        pdgid, Energy = np.array(primaries.pdgid), np.array(primaries.E)
            
        run_id, frame_index, file_id, trigger_counter = (
            np.array(file.run_id),
            np.array(file.frame_index),
            np.array(file.id),
            np.array(file.trigger_counter),
        )
        
        unique_id = create_unique_id_filetype( 
                                                pdg_id = pdgid, 
                                                energy = Energy, 
                                                is_cc_flag = is_cc_flag,
                                                run_id = run_id, 
                                                frame_index = frame_index, 
                                                evt_id = file_id
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
                                "dom_id",
                                "channel_id"
        ]

        pandas_df = hits.arrays(keys_to_extract, library="pd")
        df = pandas_df.reset_index()
        unique_extended = []
        for index in df["entry"].values:
            unique_extended.append(int(unique_id[index]))
        
        df["event_no"] = unique_extended
        
        df = remove_duplicated_event_no(df, col = 'event_no', keep = 'first')

        # keep only trigg pulses
        df = df[df["trig"] != 0]
        
        # Add the du_id
        if self.DOMs_dict is not None:
            df["du_id"] = list(map(self.DOMs_dict.get, df["dom_id"].astype(str)))

        df = df.drop(["entry", "subentry"], axis=1)
        df = creating_time_zero(df)

        return df


class KM3NeTROOTTriggPulseExtractor_detector(KM3NeTROOTExtractor):
    """Class for extracting the triggered pulse information from a file."""

    def __init__(self, name: str = "pulse_map", DOMs_dict: Optional[dict] = None):
        """Initialize the class to extract the pulse information."""
        super().__init__(name)
        self.DOMs_dict = DOMs_dict

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract pulse map information and return a dataframe.

        Args:
            file (Any): The file from which to extract triggered pulse map information.

        Returns:
            Dict[str, Any]: A dictionary containing triggered pulse map information.
        """
        pulsemap_df = self._extract_pulse_map(file)
        pulsemap_df = assert_no_uint_values(pulsemap_df)

        return pulsemap_df

    def _extract_pulse_map(self: Any, file: Any) -> pd.DataFrame:
        """Extract the pulse information and assigns unique IDs.

        Args:
            file (Any): The file from which to extract pulse information.

        Returns:
            pd.DataFrame: A dataframe containing pulse information.

        Analogous to the KM3NeTROOTPulseExtractor but doing cuts on trig.
        """
        
        unique_id = create_unique_id(
            np.array(file.run_id),
            np.array(file.frame_index),
            np.array(file.trigger_counter),
        )  # extract the unique_id

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

        # keep only trigg pulses
        df = df[df["trig"] != 0]
        
        # Add the du_id
        if self.DOMs_dict) is not None:
            df["du_id"] = list(map(self.DOMs_dict.get, df["dom_id"].astype(str)))

        df = df.drop(["entry", "subentry"], axis=1)
        df = creating_time_zero(df)

        return df
