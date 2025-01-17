"""Code with some functionalities for the extraction."""
from typing import List, Tuple, Any, Union, Literal

import numpy as np
import pandas as pd
import hashlib


def create_unique_id_filetype(
    pdg_id: List[int],
    energy: List[float],
    is_cc_flag: List[int],
    run_id: List[int],
    evt_id: List[int],
    file_type: str,
    model_hnl: str,
) -> List[str]:
    """
    Create a unique ID for each event based on particle type, energy, and other parameters.

    Parameters:
    pdg_id (List[int]): List of particle ID codes.
    energy (List[float]): List of particle energies.
    is_cc_flag (List[int]): List of flags indicating if the interaction is charged current (1) or neutral current (0).
    run_id (List[int]): List of run IDs.
    evt_id (List[int]): List of event IDs.
    file_type (str): Type of the file ('neutrino', 'muon', 'noise', 'data', 'hnl').
    model_hnl (str): Model name for HNL files. Should be none and don't be used for other file types.

    Returns:
    List[str]: List of unique IDs for each event.
    """
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
                'data': 26,
                }
    
    unique_id = []
    for i in range(len(pdg_id)):
        if file_type == 'neutrino':
            if pdg_id[i] in [12, -12, 14, -14, 16, -16]:
                if is_cc_flag[i] == 1:
                    particle = 'elec_' if abs(pdg_id[i]) == 12 else 'muon_' if abs(pdg_id[i]) == 14 else 'tau_'
                elif is_cc_flag[i] == 0:
                    particle = ''
                else:
                    raise ValueError("The is_cc_flag is not 0 or 1.")
                anti = 'anti_' if pdg_id[i] < 0 else ''
                nc = 'NC_' if is_cc_flag[i] == 0 and abs(pdg_id[i]) == 14 else ''
                if energy[i] <= 100:
                    energy_range = '1_100'
                elif (energy[i] > 100) and (energy[i] <= 500):
                    energy_range = '100_500'
                elif (energy[i] > 500) and (energy[i] <= 10000):
                    energy_range = '500_10000'
                else:
                    raise ValueError("The energy is not in the expected range [1, 10000].")
                file_id = code_dict[f'{anti}{nc}{particle}{energy_range}']
        elif file_type == 'muon':
            file_id = code_dict['atm_muon']
        elif file_type == 'noise':
            file_id = code_dict['noise']
        elif file_type == 'data':
            file_id = code_dict['data']
        elif file_type == 'hnl':
            file_id = str(abs(hash(model_hnl)) % (10 ** 3))
        else:
            raise ValueError("The file type is not recognized")
            

        unique_id.append(str(evt_id[i]) + '000' + str(run_id[i]) + '0' + str(file_id))

    return unique_id

def filter_None_NaN(
    values: Union[List[float], np.ndarray],
    padding_value: float,
) -> np.ndarray:
    """Removes None or NaN values and transforms them to padding float value."""
    values = [padding_value if v is None else v for v in values]
    values = np.array(values, dtype=float)
    values[np.isnan(values)] = padding_value
    return values

def xyz_dir_to_zen_az(
    dir_x: List[float],
    dir_y: List[float],
    dir_z: List[float],
    padding_value: float,

) -> Tuple[List[float], List[float]]:
    """Convert direction vector to zenith and azimuth angles."""
    # Compute zenith angle (elevation angle)
    with np.errstate(invalid='ignore'):
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
        else:
            pass
    return df

