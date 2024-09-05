"""Code with some functionalities for the extraction."""
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import km3pipe as kp



def create_unique_id_filetype(
    pdg_id: List[int],
    energy: List[float],
    is_cc_flag: List[int],
    run_id: List[int],
    frame_index: List[int],
    evt_id: List[int],
) -> List[str]:
    """Creating a code for each type of flavor and energy range."""
    code_dict = {'elec_1_100': 0, #TODO check if the enery ranges are suitable for ARCA
                    'elec_100_500': 1,
                    'elec_500_10000': 2,
                    'muon_1_100': 3,
                    'muon_100_500': 4,
                    'muon_500_10000': 5,
                    'tau_1_100': 6,
                    'tau_100_500': 7,
                    'tau_500_10000': 8,
                    'anti_elec_1_100': 9,
                    'anti_elec_100_500': 10,
                    'anti_elec_500_10000': 11,
                    'anti_muon_1_100': 12,
                    'anti_muon_100_500': 13,
                    'anti_muon_500_10000': 14,
                    'anti_tau_1_100': 15,
                    'anti_tau_100_500': 16,
                    'anti_tau_500_10000': 17,
                    'NC_1_100': 18,
                    'NC_100_500': 19,
                    'NC_500_10000': 20,
                    'anti_NC_1_100': 21,
                    'anti_NC_100_500': 22,
                    'anti_NC_500_10000': 23,
                    'atm_muon': 24,
                    'noise': 25,
                    'data': 26
                    }
    
    unique_id = []
    for i in range(len(pdg_id)):
        #compute the file_id
        #for electrons
        if pdg_id[i] == 12:
            if energy[i] < 100:
                file_id = code_dict['elec_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                file_id = code_dict['elec_100_500']
            else:
                file_id = code_dict['elec_500_10000']
        #for muons
        if pdg_id[i] == 14:
            if energy[i] < 100:
                if is_cc_flag[i] == 1:
                    file_id = code_dict['muon_1_100']
                else:
                    file_id = code_dict['NC_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                if is_cc_flag[i] == 1:
                    file_id = code_dict['muon_100_500']
                else:
                    file_id = code_dict['NC_100_500']
            else:
                if is_cc_flag[i] == 1:
                    file_id = code_dict['muon_500_10000']
                else:
                    file_id = code_dict['NC_500_10000']
        #for taus
        if pdg_id[i] == 16:
            if energy[i] < 100:
                file_id = code_dict['tau_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                file_id = code_dict['tau_100_500']
            else:
                file_id = code_dict['tau_500_10000']
        #for anti-electrons
        if pdg_id[i] == -12:
            if energy[i] < 100:
                file_id = code_dict['anti_elec_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                file_id = code_dict['anti_elec_100_500']
            else:
                file_id = code_dict['anti_elec_500_10000']
        #for anti-muons
        if pdg_id[i] == -14:
            if energy[i] < 100:
                if is_cc_flag[i] == 1:
                    file_id = code_dict['anti_muon_1_100']
                else:
                    file_id = code_dict['anti_NC_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                if is_cc_flag[i] == 1:
                    file_id = code_dict['anti_muon_100_500']
                else:
                    file_id = code_dict['anti_NC_100_500']
            else:
                if is_cc_flag[i] == 1:
                    file_id = code_dict['anti_muon_500_10000']
                else:
                    file_id = code_dict['anti_NC_500_10000']
        #for anti-taus
        if pdg_id[i] == -16:
            if energy[i] < 100:
                file_id = code_dict['anti_tau_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                file_id = code_dict['anti_tau_100_500']
            else:
                file_id = code_dict['anti_tau_500_10000']
        #for atmospheric muons
        if pdg_id[i] not in [12, 14, 16, -12, -14, -16, 0, 99]:
            file_id = code_dict['atm_muon']
        #for noise
        if pdg_id[i] == 0:
            file_id = code_dict['noise']
        #for data
        if pdg_id[i] == 99:
            file_id = code_dict['data']

        #compute the unique_id as evt_id + run_id + file_id + frame_index
        unique_id.append(str(evt_id[i]) + '000' + str(run_id[i]) + '0' + str(file_id))

    return unique_id





def create_unique_id(
    pdg_id: List[int],
    run_id: List[int],
    frame_index: List[int],
    trigger_counter: List[int],
) -> List[str]:
    """Create unique ID as run_id, frame_index, trigger_counter."""
    unique_id = []
    for i in range(len(pdg_id)):
        unique_id.append(
            str(run_id[i])
            + "0"
            + str(frame_index[i])
            + "0"
            + str(trigger_counter[i])
        )

    return unique_id

def create_unique_id_dbang(
    energy: List[float],
    pos_x: List[float],
    ids: List[int],
) -> List[str]:
    """Create unique ID for double bang events."""
    unique_id = []
    for i in range(len(energy)):
        unique_id.append(
            str(ids[i])
            + str(int(1000*energy[i]))
            + str(int(abs(1000*pos_x[i])))
        )
    return unique_id


def xyz_dir_to_zen_az(
    dir_x: List[float],
    dir_y: List[float],
    dir_z: List[float],
    padding_value: float,
) -> Tuple[List[float], List[float]]:
    """Convert direction vector to zenith and azimuth angles."""
    # Compute zenith angle (elevation angle)
    zenith = np.arccos(dir_z)  # zenith angle in radians

    # Compute azimuth angle
    azimuth = np.arctan2(dir_y, dir_x)  # azimuth angle in radians
    az_centered = azimuth + np.pi * np.ones(
        len(azimuth)
    )  # Center the azimuth angle around zero
    #check for NaN in the zenith and replace with padding_value
    zenith[np.isnan(zenith)] = padding_value
    #change the azimuth values to padding value if the zenith is padding value
    az_centered[zenith == padding_value] = padding_value

    return zenith, az_centered


def classifier_column_creator(
    pdgid: np.ndarray,
    is_cc_flag: List[int],
) -> Tuple[List[int], List[int]]:
    """Create helpful columns for the classifier."""
    is_muon = np.zeros(len(pdgid), dtype=int)
    is_track = np.zeros(len(pdgid), dtype=int)
    is_noise = np.zeros(len(pdgid), dtype=int)
    is_data = np.zeros(len(pdgid), dtype=int)
    
    #TODO add tau topology
    """
    primaries = f.mc_trks[:,0]
    secondaries = f.mc_trks
    
    print("%"*20)
    tau=abs(primaries.pdgid)==16
    tau_track=np.any(np.abs(secondaries.pdgid)==13,axis=1)
    result=np.logical_and(tau,tau_track)
    print(secondaries.pdgid[result])
    for i in secondaries.pdgid[result]: # this will get all track like taus, we could add a new column as 'tau_topology'
        print(i)

    """

    is_muon[abs(pdgid) == 13] = 1
    is_muon[pdgid == 81] = 1
    is_track[abs(pdgid) == 13] = 1
    is_track[pdgid == 81] = 1
    is_track[(abs(pdgid) == 14) & (is_cc_flag == 1)] = 1
    is_noise[pdgid == 0] = 1
    is_data[pdgid == 99] = 1

    return is_muon, is_track, is_noise, is_data


def creating_time_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the event time so that the first hit has zero in time."""
    df = df.sort_values(by=["event_no", "t"])
    df["min_t"] = df.groupby("event_no")["t"].transform("min")
    df["t"] = df["t"] - df["min_t"]
    df = df.drop(["min_t"], axis=1)

    return df


def assert_no_uint_values(df: pd.DataFrame) -> pd.DataFrame:
    """Assert no format no supported by sqlite is in the data."""
    for column in df.columns:
        if df[column].dtype == "uint32":
            df[column] = df[column].astype("int32")
        elif df[column].dtype == "uint64":
            df[column] = df[column].astype("int64")
    return df


def pmt_id_to_pos_dir(
        df: pd.DataFrame,
        det_file: str) -> pd.DataFrame:
    """Extract the position and direction of the hit from the pmt_id."""
    # Extract the position and direction of the hit from the pmt_id

    det = kp.hardware.Detector(det_file)


    pos_x = np.array([det.pmt_with_id(pmt_id).pos_x for pmt_id in df["pmt_id"]])
    pos_y = np.array([det.pmt_with_id(pmt_id).pos_y for pmt_id in df["pmt_id"]])
    pos_z = np.array([det.pmt_with_id(pmt_id).pos_z for pmt_id in df["pmt_id"]])
    dir_x = np.array([det.pmt_with_id(pmt_id).dir_x for pmt_id in df["pmt_id"]])
    dir_y = np.array([det.pmt_with_id(pmt_id).dir_y for pmt_id in df["pmt_id"]])
    dir_z = np.array([det.pmt_with_id(pmt_id).dir_z for pmt_id in df["pmt_id"]])

    df["pos_x"] = pos_x
    df["pos_y"] = pos_y
    df["pos_z"] = pos_z
    df["dir_x"] = dir_x
    df["dir_y"] = dir_y
    df["dir_z"] = dir_z

    return df