"""
    A merger script. 
    
    Usage:
        python3 input_dir output_dir detector truth trig_pulses pulses
    
    where:
        
        input_dir: 
            Directory with the files to merge
        
        output_dir:
            Directory where to create the merged folder with merged database.

        detector:
            Which detector was used for the creation of individual databases. 
            Options: 'ORCA' for full detector, otherwise, whatever for smaller
            configurations.

        truth:
            Name of the table used for true information.
            Options: 'truth' will include the true information, otherwise,
            either it was not included in the databases or it is not meant
            to be merged.

        trig_pulses:
            Name of the table used for true information.
            Options: 'trigg_pulse_map' will include the true information, 
            otherwise, either it was not included in the databases or it is 
            not meant to be merged.

        pulses:
            Name of the table used for true information.
            Options: 'pulse_map' will include the true information, otherwise,
            either it was not included in the databases or it is not meant
            to be merged.

"""

from graphnet.data.readers import KM3NeTROOTReader
from graphnet.data.writers import SQLiteWriter
from graphnet.data import DataConverter
from graphnet.data.extractors.km3net import (
                                                KM3NeTROOTPulseExtractorORCA, 
                                                KM3NeTROOTTriggPulseExtractorORCA, 
                                                KM3NeTROOTTruthExtractorORCA,
)
import warnings
import os
import sys
import json

# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    input_dir = sys.argv[1]          # The directory with the files to merge
    output_dir = sys.argv[2]         # The output directory 
    
    detector = sys.argv[3]
    truth = sys.argv[4]
    trig_pulses = sys.argv[5]
    pulses = sys.argv[6]

    os.makedirs(output_dir, exist_ok = True)

    extractors = []

    if detector == 'ORCA':
        if truth == 'truth':
            extractors.append(
                KM3NeTROOTTruthExtractorORCA(
                                                    name = "truth", 
                                                    DOMs_dict = DOMs_dict
                )
            )
        if trig_pulses == 'trigg_pulse_map':
            extractors.append(
                KM3NeTROOTTriggPulseExtractorORCA(
                                                        name = "trigg_pulse_map", 
                                                        DOMs_dict = DOMs_dict,                                    
                )
            )
        if pulses == 'pulse_map':
            extractors.append(
                KM3NeTROOTPulseExtractorORCA(
                                                    name = "pulse_map", 
                                                    DOMs_dict = DOMs_dict,
                                                    time_window = (-1000.0, 1000.0),
                                                    max_noise = (150, 150)
                )
            )
    else:
        if truth == 'truth':
            extractors.append(
                KM3NeTROOTTruthExtractor_detector(
                                                    name = "truth", 
                                                    DOMs_dict = DOMs_dict
                )
            )
        if trig_pulses == 'trigg_pulse_map':
            extractors.append(
                KM3NeTROOTTriggPulseExtractor_detector(
                                                        name = "trigg_pulse_map", 
                                                        DOMs_dict = DOMs_dict,                                    
                )
            )
        if pulses == 'pulse_map':
            extractors.append(
                KM3NeTROOTPulseExtractor_detector(
                                                    name = "pulse_map", 
                                                    DOMs_dict = DOMs_dict,
                                                    time_window = (-1000.0, 1000.0),
                                                    max_noise = (150, 150)
                )
            )

    # Initialize DataConverter for merging
    converter = DataConverter(
                                file_reader = KM3NeTROOTReader(),  
                                save_method = SQLiteWriter(), 
                                extractors = extractors,
                                outdir = output_dir
    )

    sqlite_files = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if i.endswith('.db')]

    # Call merge_files method to merge the databases
    converter.merge_files(files = sqlite_files)