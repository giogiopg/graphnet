from graphnet.data.readers import KM3NeTROOTReader
from graphnet.data.writers import SQLiteWriter
from graphnet.data import DataConverter
from graphnet.data.extractors.km3net import KM3NeTROOTPulseExtractor, KM3NeTROOTTriggPulseExtractor, KM3NeTROOTTruthExtractor
import warnings
import os
import sys
import json

# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    
    input_dir_file = sys.argv[1]       # The file or directory with files to process
    DOMs_dict = sys.argv[2]            # The JSON file with information about the detector
    
    with open(DOMs_dict, 'r') as f:
        DOMs_dict = json.load(f)
    
    output_dir = sys.argv[3]           # The output directory to store the db
    
    # Initialize DataConverter for merging
    converter = DataConverter(
                                file_reader = KM3NeTROOTReader(),  
                                save_method = SQLiteWriter(), 
                                extractors = [
                                                KM3NeTROOTTruthExtractor(
                                                                            name = "truth", 
                                                                            DOMs_dict = DOMs_dict
                                                ),
                                                KM3NeTROOTPulseExtractor(
                                                                            name = "pulse_map", 
                                                                            DOMs_dict = DOMs_dict,
                                                                            time_window = (-1000.0, 1000.0),
                                                                            max_noise = (150, 150)
                                                ),
                                                KM3NeTROOTTriggPulseExtractor(
                                                                                name = "trigg_pulse_map", 
                                                                                DOMs_dict = DOMs_dict,                                    
                                                ),
                                ],
                                outdir = os.path.dirname(output_dir)
    )

    # Call merge_files method to merge the databases
    if input_dir_file.endswith('.db'):
        converter._process_file( file_path = input_dir_file )
    else:
        converter( input_dir_file )
