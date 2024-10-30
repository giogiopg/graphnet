"""Code to extract the truth event information from the KM3NeT ROOT file."""

from typing import Any, Dict
import numpy as np
import pandas as pd
import km3io as ki

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    classifier_column_creator,
    create_unique_id_filetype,
    xyz_dir_to_zen_az,
    assert_no_uint_values,
    filter_None_NaN,

)
#from graphnet.data.extractors.km3net.utilities.weight_events_oscprob import compute_evt_weight



class KM3NeTROOTTruthExtractorHNL(KM3NeTROOTExtractor):
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
        nus_flavor = [12, 14, 16]
        padding_value = 99999999.0
        if len(file.mc_trks.E[0]>0):
            primaries = file.mc_trks[:, 0]
            #muon file
            if abs(np.array(primaries.pdgid)[0]) not in nus_flavor:
                # it is a muon file
                # in muon files the first entry is a 81 particle, with no physical meaning
                primaries = file.mc_trks[:, 0]
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])

                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                    padding_value,
                )
                    
                #check if if has a jmuon reconstruction
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])

                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                    padding_value,
                )



                # construct some quantities
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
                unique_id = create_unique_id_filetype(
                np.array(primaries.pdgid),
                np.array(primaries.E),
                np.ones(len(primaries.pdgid)),
                np.array(file.run_id),
                np.array(file.frame_index),
                np.array(file.id),
            )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )
                livetime= float(file.header.livetime.numberOfSeconds)
                daq = float(file.header.DAQ.livetime)

                dict_truth = {
                    "pdgid": np.array(primaries.pdgid),
                    "vrx_x": np.array(primaries.pos_x),
                    "vrx_y": np.array(primaries.pos_y),
                    "vrx_z": np.array(primaries.pos_z),
                    "zenith_nu": zen_truth,
                    "azimuth_nu": az_truth,
                    "zenith_hnl": padding_value * np.ones(len(primaries.pos_x)),
                    "azimuth_hnl": padding_value * np.ones(len(primaries.pos_x)),
                    "angle_between_showers": np.zeros(len(primaries.pos_x)),
                    "Energy_nu": np.array(primaries.E),
                    "Energy_first_shower": np.array(primaries.E),
                    "Energy_hnl": padding_value * np.ones(len(primaries.pos_x)),
                    "Energy_second_shower": padding_value * np.ones(len(primaries.pos_x)),
                    "Energy_imbalance": np.zeros(len(primaries.pos_x)),
                    "is_cc_flag": padding_value * np.ones(len(primaries.pos_x)),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    "tau_topology": padding_value * np.ones(len(primaries.pos_x)),
                    "is_hnl": np.zeros(len(primaries.pos_x)),
                    "w_osc": (daq/livetime) * np.ones(len(primaries.pos_x)),
                }

            #hnl file
            elif 5914 in file.mc_trks.pdgid[0]:
                #it is HNL file
                hnl = file.mc_trks[:, 1] #probably it will be 0, 1, 2
                first_shower = file.mc_trks[:, 2]
                second_shower = file.mc_trks[:, 4]

                # the zenith and azimuth of the light neutrino
                zen_nu, az_nu = xyz_dir_to_zen_az(
                    np.array(primaries.dir_x),
                    np.array(primaries.dir_y),
                    np.array(primaries.dir_z),
                    padding_value,
                )

                energy_imbalance = (np.array(first_shower.E) - np.array(second_shower.E)) / (np.array(first_shower.E) + np.array(second_shower.E))
                distance = np.sqrt((np.array(primaries.pos_x) - np.array(second_shower.pos_x))**2 + (np.array(primaries.pos_y) - np.array(second_shower.pos_y))**2 + (np.array(primaries.pos_z) - np.array(second_shower.pos_z))**2)

                #compute the angle between the two showers
                angle = np.arccos(np.array(first_shower.dir_x)*np.array(second_shower.dir_x) + np.array(first_shower.dir_y)*np.array(second_shower.dir_y) + np.array(first_shower.dir_z)*np.array(second_shower.dir_z))

                # the zenith and azimuth of the heavy neutrino
                zen_hnl, az_hnl = xyz_dir_to_zen_az(
                    np.array(hnl.dir_x),
                    np.array(hnl.dir_y),
                    np.array(hnl.dir_z),
                    padding_value,
                )

                livetime = float(file.header.DAQ.livetime)
                try:
                    #file.header.genvol.numberOfEvents 
                    n_gen = file.header.genvol.numberOfEvents #single header
                except:
                    n_gen = 1/file.w[:,3] #multiheader
                
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])
                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                    padding_value,
                )
                
                
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])
                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                    padding_value,
                )

                unique_id = create_unique_id_filetype(
                np.array(primaries.pdgid),
                np.array(primaries.E),
                np.ones(len(primaries.pdgid)),
                np.array(file.run_id),
                np.array(file.frame_index),
                np.array(file.id),
            )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )

                # for tau CC it is not clear what the second interaction is; 1 for shower, 2 for track, 3 for nothing
                tau_topologies = [2 if 16 in np.abs(primaries.pdgid) and 13 in np.abs(file.mc_trks.pdgid[i]) else 1 if 16 in np.abs(primaries.pdgid) else 3 for i in range(len(primaries.pdgid))]

                dict_truth = {
                    "pdgid": np.array(primaries.pdgid),
                    "vrx_x": np.array(primaries.pos_x),
                    "vrx_y": np.array(primaries.pos_y),
                    "vrx_z": np.array(primaries.pos_z),
                    "zenith_nu": zen_nu,
                    "azimuth_nu": az_nu,
                    "zenith_hnl": zen_hnl,
                    "azimuth_hnl": az_hnl,
                    "angle_between_showers": angle,
                    "Energy_nu": np.array(primaries.E),
                    "Energy_first_shower": np.array(first_shower.E),
                    "Energy_hnl": np.array(hnl.E),
                    "Energy_second_shower": np.array(second_shower.E),
                    "Energy_imbalance": energy_imbalance,
                    "distance": distance,
                    "is_cc_flag": np.zeros(len(primaries.pos_x)),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    "tau_topology": tau_topologies,
                    "is_hnl": np.ones(len(primaries.pos_x)),
                    #"w_osc": compute_evt_weight(np.array(primaries.pdgid),np.array(primaries.E),np.array(primaries.dir_z),np.array(file.w2list[:, 10] == 2),np.array(file.w[:, 1]),n_gen,livetime*np.ones(len(primaries.pos_x))),
                    "w_osc": file.w[:, 1] / n_gen,
                }

            else:
                
                # the particle is a neutrino
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

                livetime = float(file.header.DAQ.livetime)
                try:
                    file.header.genvol.numberOfEvents #single header
                except:
                    n_gen = 1/file.w[:,3] #multiheader
                
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                
                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])
                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                    padding_value,
                )
                
                
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])
                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                    padding_value,
                )

                unique_id = create_unique_id_filetype(
                np.array(primaries.pdgid),
                np.array(primaries.E),
                np.ones(len(primaries.pdgid)),
                np.array(file.run_id),
                np.array(file.frame_index),
                np.array(file.id),
            )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )

                # for tau CC it is not clear what the second interaction is; 1 for shower, 2 for track, 3 for nothing
                tau_topologies = [2 if 16 in np.abs(primaries.pdgid) and 13 in np.abs(file.mc_trks.pdgid[i]) else 1 if 16 in np.abs(primaries.pdgid) else 3 for i in range(len(primaries.pdgid))]

                dict_truth = {
                    "pdgid": np.array(primaries.pdgid),
                    "vrx_x": np.array(primaries.pos_x),
                    "vrx_y": np.array(primaries.pos_y),
                    "vrx_z": np.array(primaries.pos_z),
                    "zenith_nu": zen_truth,
                    "azimuth_nu": az_truth,
                    "zenith_hnl": padding_value * np.ones(len(primaries.pos_x)),
                    "azimuth_hnl": padding_value * np.ones(len(primaries.pos_x)),
                    "angle_between_showers": np.zeros(len(primaries.pos_x)),
                    "Energy_nu": np.array(primaries.E),
                    "Energy_first_shower": np.array(primaries.E),
                    "Energy_hnl": padding_value * np.ones(len(primaries.pos_x)),
                    "Energy_second_shower": padding_value * np.ones(len(primaries.pos_x)),
                    "Energy_imbalance": np.zeros(len(primaries.pos_x)),
                    "distance": np.zeros(len(primaries.pos_x)),
                    "is_cc_flag": np.array(file.w2list[:, 10] == 2),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    "tau_topology": tau_topologies ,
                    "is_hnl": np.zeros(len(primaries.pos_x)),
                    #"w_osc": compute_evt_weight(np.array(primaries.pdgid),np.array(primaries.E),np.array(primaries.dir_z),np.array(file.w2list[:, 10] == 2),np.array(file.w[:, 1]),n_gen,livetime*np.ones(len(primaries.pos_x))),
                    "w_osc": padding_value * np.ones(len(primaries.pos_x)),
                }
        else:

            if file.header['calibration']=="dynamical": #data file
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])
                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                    padding_value,
                )
                    
                #check if if has a jmuon reconstruction
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])
                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                    padding_value,
                )



                # construct some quantities
                zen_truth, az_truth = padding_value * np.ones(len(primaries_jmuon.E)), padding_value * np.ones(len(primaries_jmuon.E))
                part_dir_x, part_dir_y, part_dir_z = padding_value * np.ones(len(primaries_jmuon.E)), padding_value * np.ones(len(primaries_jmuon.E)),padding_value * np.ones(len(primaries_jmuon.E))
                unique_id = create_unique_id_filetype(
                    26 * np.ones(len(file.run_id)),
                    np.ones(len(file.run_id)),
                    np.ones(len(file.run_id)),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
            )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )
                daq = float(file.header.DAQ.livetime)
                livetime= float(file.header.DAQ.livetime)

                dict_truth = {
                    "pdgid": 99 * np.ones(len(primaries_jmuon.E),dtype=int),
                    "vrx_x": padding_value * np.ones(len(primaries_jmuon.E)),
                    "vrx_y": padding_value * np.ones(len(primaries_jmuon.E)),
                    "vrx_z": padding_value * np.ones(len(primaries_jmuon.E)),
                    "zenith_nu": padding_value * np.ones(len(primaries_jmuon.E)),
                    "azimuth_nu": padding_value * np.ones(len(primaries_jmuon.E)),
                    "zenith_hnl": padding_value * np.ones(len(primaries_jmuon.E)),
                    "azimuth_hnl": padding_value * np.ones(len(primaries_jmuon.E)),
                    "angle_between_showers": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Energy_nu": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Energy_first_shower": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Energy_hnl": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Energy_second_shower": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Energy_imbalance": padding_value * np.ones(len(primaries_jmuon.E)),
                    "distance": padding_value * np.ones(len(primaries_jmuon.E)),
                    "is_cc_flag": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    "tau_topology": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "is_hnl": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "w_osc": np.ones(len(primaries_jmuon.pos_x)),
                }

            else:
                # the event is pure noise
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])

                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                    padding_value,
                )
                    
                #check if if has a jmuon reconstruction
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])

                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                    padding_value,
                )



                # construct some quantities
                zen_truth, az_truth = padding_value * np.ones(len(primaries_jmuon.E)), padding_value * np.ones(len(primaries_jmuon.E))
                part_dir_x, part_dir_y, part_dir_z = padding_value * np.ones(len(primaries_jmuon.E)), padding_value * np.ones(len(primaries_jmuon.E)),padding_value * np.ones(len(primaries_jmuon.E))
                unique_id = create_unique_id_filetype(
                    np.zeros(len(file.run_id)),
                    np.ones(len(file.run_id)),
                    np.ones(len(file.run_id)),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
            )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )

                daq = float(file.header.DAQ.livetime)
                livetime= float(file.header.K40) # simulated noise

                dict_truth = {
                    "pdgid": np.zeros(len(primaries_jmuon.E),dtype=int),
                    "vrx_x": padding_value * np.ones(len(primaries_jmuon.E)),
                    "vrx_y": padding_value * np.ones(len(primaries_jmuon.E)),
                    "vrx_z": padding_value * np.ones(len(primaries_jmuon.E)),
                    "zenith": padding_value * np.ones(len(primaries_jmuon.E)),
                    "azimuth": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_x": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_y": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_z": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Energy": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Bj_x": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "Bj_y": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "is_cc_flag": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    "tau_topology": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "w_osc": (daq/livetime) * np.ones(len(primaries_jmuon.pos_x)),
                }

                

        truth_df = pd.DataFrame(dict_truth)
        is_muon, is_track, is_noise, is_data = classifier_column_creator(

            np.array(dict_truth["pdgid"]), np.array(dict_truth["is_cc_flag"])
        )
        truth_df["is_muon"] = is_muon
        truth_df["is_track"] = is_track
        truth_df["is_noise"] = is_noise
        truth_df["is_data"] = is_data
            

        return truth_df

