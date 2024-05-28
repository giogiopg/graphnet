from graphnet.data.readers import KM3NeTROOTReader
from graphnet.data.writers import SQLiteWriter
from graphnet.data import DataConverter
from graphnet.data.extractors.km3net import KM3NeTROOTTruthExtractor, KM3NeTROOTTriggPulseExtractor
import warnings
import os
import sys

# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok = True)

    # Initialize DataConverter for merging
    converter = DataConverter(
                                file_reader = KM3NeTROOTReader(),  
                                save_method = SQLiteWriter(), 
                                extractors = [
                                                KM3NeTROOTTruthExtractor(name = "truth"),
                                                KM3NeTROOTTriggPulseExtractor(name = "trigg_pulse_map")
                                ],
                                outdir = output_dir
    )

    sqlite_files = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if i.endswith('.db')]

    # Call merge_files method to merge the databases
    converter.merge_files(files = sqlite_files)
