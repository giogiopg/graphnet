"""Code to extract the truth event information from the KM3NeT ROOT file."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import km3io as ki
import awkward as ak
import math

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    classifier_column_creator,
    create_unique_id,
    create_unique_id_filetype,
    filter_None_NaN,
    xyz_dir_to_zen_az,
    assert_no_uint_values,
    remove_duplicated_event_no,
)

class KM3NeTROOTTruthExtractorORCA(KM3NeTROOTExtractor):
    """Class for extracting the truth information from a file."""

    def __init__(self, name: str = "truth", DOMs_dict: Optional[dict] = None):
        """Initialize the class to extract the truth information."""
        super().__init__(name)
        self.DOMs_dict = DOMs_dict

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
        nus_flavor = [12, 14, 16]
        
        n_evts = len(file)
        
        padding_value = 999.0 
        ones_vector = np.ones(n_evts)
        padding_vector = padding_value * ones_vector
       
        hits = file.hits
        trig_hits = hits[hits.trig != 0]
        
        trig_hits_dom_id = trig_hits.dom_id
        
        n_trig_hits = ak.count(trig_hits_dom_id, axis = 1)
        
        if self.DOMs_dict is not None:
            unique_doms = []
            unique_dus = []
            for evt in range(len(file)):
                # This loop calculate how many DOMs are triggered during an event
                dom_array = np.array(trig_hits_dom_id[evt])
                unique_doms.append( len(np.unique(dom_array)))
                unique_dus.append( len(np.unique(np.array(list(map(self.DOMs_dict.get, dom_array.astype(str)))) )) )
            
            unique_doms = np.array(unique_doms)
            unique_dus = np.array(unique_dus)
        else:
            unique_doms = padding_vector
            unique_dus = padding_vector

        try:
            if abs(np.array(file.mc_trks[0, 0].pdgid)) not in nus_flavor:
                # Muon file
                primaries = file.mc_trks[:, 1]

                w2_gseagen_ps = n_gen = Bj_x = Bj_y = i_chan = is_cc_flag = padding_vector
                
                DAQ = float(file.header.DAQ.livetime) * ones_vector # Time of measurement in water
                
            elif abs(np.array(file.mc_trks[0, 0].pdgid)) in nus_flavor:
                # Neutrino file
                primaries = file.mc_trks[:, 0]

                w2_gseagen_ps = np.array(file.w2list[:, 0])
                n_gen = float(file.header.genvol.numberOfEvents) * ones_vector
                Bj_x = np.array(file.w2list[:, 7])
                Bj_y = np.array(file.w2list[:, 8])
                i_chan = np.array(file.w2list[:, 9])
                is_cc_flag = np.array(file.w2list[:, 10] == 2)
                
                DAQ = float(file.header.DAQ.livetime) * ones_vector # Time of measurement in water
                               
            # construct some quantities
            pdgid, Energy = np.array(primaries.pdgid), np.array(primaries.E)
            vrx_x, vrx_y, vrx_z = np.array(primaries.pos_x), np.array(primaries.pos_y), np.array(primaries.pos_z)
            part_dir_x, part_dir_y, part_dir_z = np.array(primaries.dir_x), np.array(primaries.dir_y), np.array(primaries.dir_z)
            zen_truth, az_truth = xyz_dir_to_zen_az( part_dir_x, part_dir_y, part_dir_z )
            
        except ValueError:
            pass
            
        jshower_reco = ki.tools.best_jshower(file.trks)
        jmuon_reco = ki.tools.best_jmuon(file.trks)
            
        # Check if if has a jshower reconstruction
        if jshower_reco.E[0] is not None:
            jshower_reco_E = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.E])
            jshower_reco_pos_x = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.pos_x])
            jshower_reco_pos_y = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.pos_y])
            jshower_reco_pos_z = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.pos_z])
            jshower_reco_dir_x = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.dir_x])
            jshower_reco_dir_y = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.dir_y])
            jshower_reco_dir_z = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.dir_z])
            jshower_reco_lik = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.lik])
            zen_jshower, az_jshower = xyz_dir_to_zen_az( jshower_reco_dir_x, jshower_reco_dir_y, jshower_reco_dir_z, )
        else:
            jshower_reco_E = jshower_reco_pos_x = jshower_reco_pos_y = jshower_reco_pos_z = jshower_reco_lik = zen_jshower = az_jshower = padding_vector

        # Check if if has a jmuon reconstruction
        if jmuon_reco.E[0] is not None:
            jmuon_reco_E = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.E])
            jmuon_reco_pos_x = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.pos_x])
            jmuon_reco_pos_y = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.pos_y])
            jmuon_reco_pos_z = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.pos_z])
            jmuon_reco_dir_x = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.dir_x])
            jmuon_reco_dir_y = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.dir_y])
            jmuon_reco_dir_z = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.dir_z])
            jmuon_reco_lik = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.lik])
            zen_jmuon, az_jmuon = xyz_dir_to_zen_az( jmuon_reco_dir_x, jmuon_reco_dir_y, jmuon_reco_dir_z, )
        else:
            jmuon_reco_E = jmuon_reco_pos_x = jmuon_reco_pos_y = jmuon_reco_pos_z = jmuon_reco_lik = zen_jmuon = az_jmuon = padding_vector

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
        
        dict_truth = {
                        "pdgid": pdgid,
                        "vrx_x": vrx_x,
                        "vrx_y": vrx_y,
                        "vrx_z": vrx_z,
                        "zenith": zen_truth,
                        "azimuth": az_truth,
                        "part_dir_x": part_dir_x,
                        "part_dir_y": part_dir_y,
                        "part_dir_z": part_dir_z,
                        "Energy": Energy,
                        "Bj_x": Bj_x,
                        "Bj_y": Bj_y,
                        "i_chan": i_chan,
                        "is_cc_flag": is_cc_flag,
                        "jshower_E": jshower_reco_E,
                        "jshower_lik": jshower_reco_lik,
                        "jshower_pos_x": jshower_reco_pos_x,
                        "jshower_pos_y": jshower_reco_pos_y,
                        "jshower_pos_z": jshower_reco_pos_z,
                        "jshower_zenith": zen_jshower,
                        "jshower_azimuth": az_jshower,
                        "jmuon_E": jmuon_reco_E,
                        "jmuon_lik": jmuon_reco_lik,
                        "jmuon_pos_x": jmuon_reco_pos_x,
                        "jmuon_pos_y": jmuon_reco_pos_y,
                        "jmuon_pos_z": jmuon_reco_pos_z,
                        "jmuon_zenith": zen_jmuon,
                        "jmuon_azimuth": az_jmuon,
                        "n_hits": np.array(file.n_hits),
                        "n_trig_hits": np.array(n_trig_hits),
                        "w2_gseagen_ps": w2_gseagen_ps,
                        "DAQ": DAQ,
                        "livetime": DAQ,
                        "n_gen": n_gen,
                        "run_id": run_id,
                        "frame_index": frame_index,
                        "trigger_counter": trigger_counter,
                        "event_no": np.array(unique_id).astype(int),
        }

        truth_df = pd.DataFrame(dict_truth)
        is_muon, is_track = classifier_column_creator(
            np.array(dict_truth["pdgid"]), np.array(dict_truth["is_cc_flag"])
        )
        truth_df["is_muon"] = is_muon
        truth_df["is_track"] = is_track
        
        # Information about the number of DOMs and DUs triggered
        truth_df["unique_doms"] = unique_doms.astype(int)
        truth_df["unique_dus"] = unique_dus.astype(int)
        
        truth_df = remove_duplicated_event_no(truth_df, col = 'event_no', keep = 'first')

        return truth_df

class KM3NeTROOTTruthExtractor_detector(KM3NeTROOTExtractor):
    """Class for extracting the truth information from a file."""

    def __init__(self, name: str = "truth", DOMs_dict: Optional[dict] = None):
        """Initialize the class to extract the truth information."""
        super().__init__(name)
        self.DOMs_dict = DOMs_dict

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
        nus_flavor = [12, 14, 16]
        
        n_evts = len(file)
        
        padding_value = 999.0 
        ones_vector = np.ones(n_evts)
        padding_vector = padding_value * ones_vector
       
        hits = file.hits
        trig_hits = hits[hits.trig != 0]
        
        trig_hits_dom_id = trig_hits.dom_id
        
        n_trig_hits = ak.count(trig_hits_dom_id, axis = 1)
        
        if self.DOMs_dict is not None:
            unique_doms = []
            unique_dus = []
            for evt in range(len(file)):
                # This loop calculate how many DOMs are triggered during an event
                dom_array = np.array(trig_hits_dom_id[evt])
                unique_doms.append( len(np.unique(dom_array)))
                unique_dus.append( len(np.unique(np.array(list(map(self.DOMs_dict.get, dom_array.astype(str)))) )) )
            
            unique_doms = np.array(unique_doms)
            unique_dus = np.array(unique_dus)
        else:
            unique_doms = padding_vector
            unique_dus = padding_vector

        try:
            if abs(np.array(file.mc_trks[0, 0].pdgid)) not in nus_flavor:
                # Muon file
                primaries = file.mc_trks[:, 1]

                w2_gseagen_ps = n_gen = Bj_x = Bj_y = i_chan = is_cc_flag = padding_vector
                
                livetime = float(file.header.livetime.numberOfSeconds) * ones_vector # Simulated time
                DAQ = float(file.header.DAQ.livetime) * ones_vector # Time of measurement in water
                
            elif abs(np.array(file.mc_trks[0, 0].pdgid)) in nus_flavor:
                # Neutrino file
                primaries = file.mc_trks[:, 0]

                w2_gseagen_ps = np.array(file.w2list[:, 0])
                n_gen = np.array(1/file.w[:,3])
                Bj_x = np.array(file.w2list[:, 7])
                Bj_y = np.array(file.w2list[:, 8])
                i_chan = np.array(file.w2list[:, 9])
                is_cc_flag = np.array(file.w2list[:, 10] == 2)
                
                livetime = float(file.header.DAQ.livetime) * ones_vector # Simulated time
                DAQ = float(file.header.DAQ.livetime) * ones_vector # Time of measurement in water
                               
            # construct some quantities
            pdgid, Energy = np.array(primaries.pdgid), np.array(primaries.E)
            vrx_x, vrx_y, vrx_z = np.array(primaries.pos_x), np.array(primaries.pos_y), np.array(primaries.pos_z)
            part_dir_x, part_dir_y, part_dir_z = np.array(primaries.dir_x), np.array(primaries.dir_y), np.array(primaries.dir_z)
            zen_truth, az_truth = xyz_dir_to_zen_az( part_dir_x, part_dir_y, part_dir_z )
            
        except ValueError:
            
            try: # Noise file
                livetime = float(file.header.K40) * ones_vector # Simulated time
                DAQ = float(file.header.DAQ.livetime) * ones_vector # Time of measurement in water
            
            except AttributeError:
                livetime = float(file.header.DAQ.livetime) * ones_vector # Simulated time
                DAQ = float(file.header.DAQ.livetime) * ones_vector # Time of measurement in water
            
            w2_gseagen_ps = n_gen = Bj_x = Bj_y = i_chan = is_cc_flag = padding_vector
            
            # construct some quantities
            pdgid = Energy = padding_vector
            vrx_x = vrx_y = vrx_z = padding_vector
            part_dir_x = part_dir_y = part_dir_z = padding_vector
            zen_truth = az_truth = padding_vector
            
        jshower_reco = ki.tools.best_jshower(file.trks)
        jmuon_reco = ki.tools.best_jmuon(file.trks)
            
        # Check if if has a jshower reconstruction
        if jshower_reco.E[0] is not None:
            jshower_reco_E = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.E])
            jshower_reco_pos_x = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.pos_x])
            jshower_reco_pos_y = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.pos_y])
            jshower_reco_pos_z = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.pos_z])
            jshower_reco_dir_x = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.dir_x])
            jshower_reco_dir_y = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.dir_y])
            jshower_reco_dir_z = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.dir_z])
            jshower_reco_lik = np.array([filter_None_NaN(element, padding_value) for element in jshower_reco.lik])
            zen_jshower, az_jshower = xyz_dir_to_zen_az( jshower_reco_dir_x, jshower_reco_dir_y, jshower_reco_dir_z, )
        else:
            jshower_reco_E = jshower_reco_pos_x = jshower_reco_pos_y = jshower_reco_pos_z = jshower_reco_lik = zen_jshower = az_jshower = padding_vector

        # Check if if has a jmuon reconstruction
        if jmuon_reco.E[0] is not None:
            jmuon_reco_E = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.E])
            jmuon_reco_pos_x = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.pos_x])
            jmuon_reco_pos_y = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.pos_y])
            jmuon_reco_pos_z = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.pos_z])
            jmuon_reco_dir_x = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.dir_x])
            jmuon_reco_dir_y = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.dir_y])
            jmuon_reco_dir_z = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.dir_z])
            jmuon_reco_lik = np.array([filter_None_NaN(element, padding_value) for element in jmuon_reco.lik])
            zen_jmuon, az_jmuon = xyz_dir_to_zen_az( jmuon_reco_dir_x, jmuon_reco_dir_y, jmuon_reco_dir_z, )
        else:
            jmuon_reco_E = jmuon_reco_pos_x = jmuon_reco_pos_y = jmuon_reco_pos_z = jmuon_reco_lik = zen_jmuon = az_jmuon = padding_vector

        run_id, frame_index, trigger_counter = (
            np.array(file.run_id),
            np.array(file.frame_index),
            np.array(file.trigger_counter),
        )
        
        unique_id = create_unique_id( run_id, frame_index, trigger_counter, )
        
        dict_truth = {
                        "pdgid": pdgid,
                        "vrx_x": vrx_x,
                        "vrx_y": vrx_y,
                        "vrx_z": vrx_z,
                        "zenith": zen_truth,
                        "azimuth": az_truth,
                        "part_dir_x": part_dir_x,
                        "part_dir_y": part_dir_y,
                        "part_dir_z": part_dir_z,
                        "Energy": Energy,
                        "Bj_x": Bj_x,
                        "Bj_y": Bj_y,
                        "i_chan": i_chan,
                        "is_cc_flag": is_cc_flag,
                        "jshower_E": jshower_reco_E,
                        "jshower_lik": jshower_reco_lik,
                        "jshower_pos_x": jshower_reco_pos_x,
                        "jshower_pos_y": jshower_reco_pos_y,
                        "jshower_pos_z": jshower_reco_pos_z,
                        "jshower_zenith": zen_jshower,
                        "jshower_azimuth": az_jshower,
                        "jmuon_E": jmuon_reco_E,
                        "jmuon_lik": jmuon_reco_lik,
                        "jmuon_pos_x": jmuon_reco_pos_x,
                        "jmuon_pos_y": jmuon_reco_pos_y,
                        "jmuon_pos_z": jmuon_reco_pos_z,
                        "jmuon_zenith": zen_jmuon,
                        "jmuon_azimuth": az_jmuon,
                        "n_hits": np.array(file.n_hits),
                        "n_trig_hits": np.array(n_trig_hits),
                        "w2_gseagen_ps": w2_gseagen_ps,
                        "DAQ": DAQ,
                        "livetime": livetime,
                        "n_gen": n_gen,
                        "run_id": run_id,
                        "frame_index": frame_index,
                        "trigger_counter": trigger_counter,
                        "event_no": np.array(unique_id).astype(int),
        }

        truth_df = pd.DataFrame(dict_truth)
        is_muon, is_track = classifier_column_creator(
            np.array(dict_truth["pdgid"]), np.array(dict_truth["is_cc_flag"])
        )
        truth_df["is_muon"] = is_muon
        truth_df["is_track"] = is_track
        
        # Information about the number of DOMs and DUs triggered
        truth_df["unique_doms"] = unique_doms.astype(int)
        truth_df["unique_dus"] = unique_dus.astype(int)
        
        truth_df = remove_duplicated_event_no(truth_df, col = 'event_no', keep = 'first')

        return truth_df