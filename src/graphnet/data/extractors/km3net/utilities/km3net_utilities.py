"""Code with some functionalities for the extraction."""
from typing import List, Tuple, Any, Union, Literal

import numpy as np
import pandas as pd
import math

def create_unique_id_filetype(
    pdg_id: List[int],
    energy: List[float],
    is_cc_flag: List[int],
    run_id: List[int],
) -> List[str]:
    """Creating a code for each type of flavor and energy range."""
    
    code_dict = {
                    (12, 1, 0) : ('elec_1_100', 0),
                    (12, 1, 1) : ('elec_100_500', 1),
                    (12, 1, 2) : ('elec_500_10000', 2),
                    (14, 1, 0) : ('muon_1_100', 3),
                    (14, 1, 1) : ('muon_100_500', 4),
                    (14, 1, 2) : ('muon_500_10000', 5),
                    (16, 1, 0) : ('tau_1_100', 6),
                    (16, 1, 1) : ('tau_100_500', 7),
                    (16, 1, 2) : ('tau_500_10000', 8),
                    (-12, 1, 0) : ('anti_elec_1_100', 9),
                    (-12, 1, 1) : ('anti_elec_100_500', 10),
                    (-12, 1, 2) : ('anti_elec_500_10000', 11),
                    (-14, 1, 0) : ('anti_muon_1_100', 12),
                    (-14, 1, 1) : ('anti_muon_100_500', 13),
                    (-14, 1, 2) : ('anti_muon_500_10000', 14),
                    (-16, 1, 0) : ('anti_tau_1_100', 15),
                    (-16, 1, 1) : ('anti_tau_100_500', 16),
                    (-16, 1, 2) : ('anti_tau_500_10000', 17),
                    (12, 2, 0) : ('NC_1_100', 18),
                    (12, 2, 1) : ('NC_100_500', 19),
                    (12, 2, 2) : ('NC_500_10000', 20),
                    (-12, 2, 0) : ('anti_NC_1_100', 21),
                    (-12, 2, 1) : ('anti_NC_100_500', 22),
                    (-12, 2, 2) : ('anti_NC_500_10000', 23),
                    (13) : ('atm_muon', 24)
    }
    
    unique_id = []
    
    for i in range(len(pdg_id)):
        keys = []
        
        keys.append(pdg_id[i])
        
        if pdg_id[i] != 13:
            keys.append(is_cc_flag[i])
            if energy[i] < 100:
                keys.append(0)
            elif (energy[i] > 100) or (energy[i] < 500):
                keys.append(1)
            elif energy[i] > 500:
                keys.append(2)
        
        file_id = code_dict[tuple(keys)][1]
        
        #unique_id.append(str(evt_id[i]) + '000' + str(run_id[i]) + '0' + str(file_id))
        unique_id.append(str(file_id) + '0' + str(run_id[i]) + '0' + str(i))

    return unique_id

    return unique_id

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

def filter_None_NaN(
    value: Union[float, None, Literal[math.nan]],
    padding_value: float,
) -> float:
    """Removes None or Nan values and transforms it to padding float value."""
    if value is None or math.isnan(value):
        return padding_value
    else:
        return value

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

    is_muon[pdgid == 13] = 1
    is_track[pdgid == 13] = 1
    is_track[(abs(pdgid) == 14) & (is_cc_flag == 1)] = 1

    return is_muon, is_track

def creating_time_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the event time so that the first hit has zero in time."""
    
    df = df.sort_values(by=["event_no", "t"])
    df["min_t"] = df.groupby("event_no")["t"].transform("min")
    df["t"] = df["t"] - df["min_t"]
    df = df.drop(["min_t"], axis=1)

    return df

def creating_time_zero_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the event time so that the first hit has zero in time."""
    
    # Sort the dataframe to ensure proper ordering
    df = df.sort_values(by=["event_no", "t", "trig"])

    # Create a new column for the time of trig pulses
    df["t trig"] = df["t"].where(df["trig"] != 0, 0)

    # Calculate the mean time of trig pulses per event_no, considering only rows where trig != 0
    min_trig = df[df["trig"] != 0].groupby("event_no")["t"].min()

    # Map the mean values back to the original DataFrame
    df["t min"] = df["event_no"].map(min_trig)

    # Center the time of the events to the mean of trig pulses
    df["t"] = df["t"] - df["t min"]
    
    df = df.drop(["t min", "t trig"], axis=1)

    return df

def assert_no_uint_values(df: pd.DataFrame) -> pd.DataFrame:
    """Assert no format no supported by sqlite is in the data."""
    for column in df.columns:
        if df[column].dtype == "uint32":
            df[column] = df[column].astype("int32")
        elif df[column].dtype == "uint64":
            df[column] = df[column].astype("int64")
    return df

def remove_duplicated_event_no(
    df: pd.DataFrame, 
    col: Union[str, List[str]] = 'event_no', 
    keep: str = 'first'
) -> pd.DataFrame:
    """When the event_no is repeated, keep only one of them."""
    return df.drop_duplicates(subset = col, keep = keep)

def noise_selection(
                        df: pd.DataFrame, 
                        time_window: Tuple[float, float] = (-1000.0, 1000.0), 
                        max_noise: Tuple[int, int] = (150, 150)
) -> pd.DataFrame:
    
    """Do a selection of noise pulses based on centered time for each event_no group."""
    
    # Sort the dataframe to ensure proper ordering
    df = df.sort_values(by=["event_no", "t", "trig"])

    # Create a new column for the time of trig pulses
    df["t trig"] = df["t"].where(df["trig"] != 0, 0)

    # Calculate the mean time of trig pulses per event_no, considering only rows where trig != 0
    mean_trig = df[df["trig"] != 0].groupby("event_no")["t"].mean()

    # Map the mean values back to the original DataFrame
    df["t mean"] = df["event_no"].map(mean_trig)

    # Center the time of the events to the mean of trig pulses
    df["t center"] = df["t"] - df["t mean"]
    
    # Sort the DataFrame by event_no and t center for trig pulses
    df_trig = df[df["trig"] != 0].sort_values(by=["event_no", "t center"])

    # Define mask for noise pulses (trig == 0)
    mask_noise = df["trig"] == 0

    # Select noise pulses with time_window[0] < t center < 0, limited to max_noise[0] per event_no
    mask_low_time_window = (df["t center"] > time_window[0]) & (df["t center"] < 0)
    df_noise_low = df[mask_noise & mask_low_time_window].groupby('event_no', as_index=False).head(max_noise[0])

    # Select noise pulses with 0 < t center < time_window[1], limited to max_noise[1] per event_no
    mask_high_time_window = (df["t center"] > 0) & (df["t center"] < time_window[1])
    df_noise_high = df[mask_noise & mask_high_time_window].groupby('event_no', as_index=False).head(max_noise[1])
    
    # Concatenate all selected rows (trig pulses + noise pulses)
    df = pd.concat([df_trig, df_noise_low, df_noise_high])

    # Sort by event_no and original time t
    df = df.sort_values(by=["event_no", "t"])

    # Drop the temporary columns used for calculations (make sure these columns exist)
    df = df.drop(["t center", "t mean", "t trig"], axis=1)
    
    return df