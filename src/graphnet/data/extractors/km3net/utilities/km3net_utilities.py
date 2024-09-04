"""Code with some functionalities for the extraction."""
from typing import List, Tuple, Any

import numpy as np
import pandas as pd


def create_unique_id(
    run_id: List[int],
    frame_index: List[int],
    trigger_counter: List[int],
) -> List[str]:
    """Create unique ID as run_id, frame_index, trigger_counter."""
    unique_id = []
    for i in range(len(run_id)):
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
) -> Tuple[List[float], List[float]]:
    """Convert direction vector to zenith and azimuth angles."""
    # Compute zenith angle (elevation angle)
    zenith = np.arccos(dir_z)  # zenith angle in radians

    # Compute azimuth angle
    azimuth = np.arctan2(dir_y, dir_x)  # azimuth angle in radians
    az_centered = azimuth + np.pi * np.ones(
        len(azimuth)
    )  # Center the azimuth angle around zero

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
    #df = df.sort_values(by=["event_no", "t"])
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

def mask_saturated_pmts(df: pd.DataFrame, tot:int = 254) -> pd.DataFrame:
    """Mask saturated PMTs in each event independently"""
    condition_rows = df[df['tot'] > tot][['event_no', 'channel_id', 'dom_id']]
    merged_df = df.merge(condition_rows, on=['event_no', 'channel_id', 'dom_id'], how='left', indicator=True)
    filtered_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    return filtered_df