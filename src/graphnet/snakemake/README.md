# **Instructions on how to process data with snakemake**

Activate the virtual environment:
```
conda activate graphnet
```
Now, install the following dependency:
```
conda install -c conda-forge apptainer=1.1.5
```
Then, create a symbolic link from singularity to apptainer 
```
cd /graphnet/bin/
ln -s apptainer singularity
```
Install the following packages:

```
pip install snakemake==7.32.4
pip install pulp==2.7.0
```

To process data, do for instance:
```
snakemake -s {snakefile} --use-singularity --configfile {config.yaml} -c {cores} -d {destination_folder} -k -np
```
The flags have the following meaning:
- `-s {snakefile}`: points to the snakefile where you have defined the workflow with the correspondant rules.
- `--use-singularity`: the workflow is run inside containers.
- `--configfile {config.yaml}`: points to the configuration file with all the necessary options to run the workflow.
- `-c {cores}`: to specify the number of cores to use.
- `-d {destination_folder}`: all the outputs of the rules and the logs/benchmark files will be stored in that foder.
- `-k` or `--keep-going`: the workflow will contain even if it fails for a file.
- `-n`: does a dry-run, it will print the jobs that would be executed without actually executing them allowing you to check that everything is set up correctly before running it for real.
- `-p`: print the shell commands as it would be executed for each job. It helps to visualize what commands will be run during the workflow.

An example of how `{destination_folder}` will look like with the examples given here is:
```
test
├── benchmark
│   ├── downloads
│   │   ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010873.jorcarec.aanet.2421.tsv
│   │   ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010874.jorcarec.aanet.2422.tsv
│   │   ├── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010873.jorcarec.aanet.2421.tsv
│   │   └── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010874.jorcarec.aanet.2422.tsv
│   └── SQLite
│       ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010873.jorcarec.aanet.2421.tsv
│       ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010874.jorcarec.aanet.2422.tsv
│       ├── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010873.jorcarec.aanet.2421.tsv
│       └── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010874.jorcarec.aanet.2422.tsv
├── iRods
│   ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010873.jorcarec.aanet.2421.root
│   ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010874.jorcarec.aanet.2422.root
│   ├── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010873.jorcarec.aanet.2421.root
│   └── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010874.jorcarec.aanet.2422.root
├── logs
│   ├── downloads
│   │   ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010873.jorcarec.aanet.2421.log
│   │   ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010874.jorcarec.aanet.2422.log
│   │   ├── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010873.jorcarec.aanet.2421.log
│   │   └── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010874.jorcarec.aanet.2422.log
│   ├── graphnet_20240515-155520.log
│   ├── graphnet_20240515-155532.log
│   ├── graphnet_20240515-155713.log
│   ├── graphnet_20240515-155734.log
│   └── SQLite
│       ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010873.jorcarec.aanet.2421.log
│       ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010874.jorcarec.aanet.2422.log
│       ├── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010873.jorcarec.aanet.2421.log
│       └── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010874.jorcarec.aanet.2422.log
└── SQLite
    ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010873.jorcarec.aanet.2421.db
    ├── mcv7.1_nn_training.gsg_elec-CC_100-500GeV.sirene.jterbr00010874.jorcarec.aanet.2422.db
    ├── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010873.jorcarec.aanet.2421.db
    └── mcv7.1_nn_training.gsg_tau-CC_3-100GeV.km3sim.jterbr00010874.jorcarec.aanet.2422.db
```
For compressed files:
```
test_compressed/
├── benchmark
│   ├── downloads
│   │   └── gsg_tau-CC_3.0-100.0GeV.km3sim.jte.jmergefit.offline_81-100.tsv
│   ├── extractions
│   │   └── gsg_tau-CC_3.0-100.0GeV.km3sim.jte.jmergefit.offline_81-100.tsv
│   └── SQLite
│       ├── gsg_tau-CC_3.0-100.0GeV.100.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.81.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.82.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.83.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.84.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.85.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.86.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.87.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.88.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.89.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.90.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.91.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.92.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.93.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.94.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.95.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.96.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.97.km3sim.jte.jmergefit.offline.tsv
│       ├── gsg_tau-CC_3.0-100.0GeV.98.km3sim.jte.jmergefit.offline.tsv
│       └── gsg_tau-CC_3.0-100.0GeV.99.km3sim.jte.jmergefit.offline.tsv
├── iRods
│   ├── gsg_tau-CC_3.0-100.0GeV.km3sim.jte.jmergefit.offline_81-100
│   │   ├── gsg_tau-CC_3.0-100.0GeV.100.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.81.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.82.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.83.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.84.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.85.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.86.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.87.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.88.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.89.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.90.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.91.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.92.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.93.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.94.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.95.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.96.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.97.km3sim.jte.jmergefit.offline.root
│   │   ├── gsg_tau-CC_3.0-100.0GeV.98.km3sim.jte.jmergefit.offline.root
│   │   └── gsg_tau-CC_3.0-100.0GeV.99.km3sim.jte.jmergefit.offline.root
│   └── gsg_tau-CC_3.0-100.0GeV.km3sim.jte.jmergefit.offline_81-100.tar.gz
├── logs
│   ├── downloads
│   │   └── gsg_tau-CC_3.0-100.0GeV.km3sim.jte.jmergefit.offline_81-100.log
│   ├── extractions
│   │   └── gsg_tau-CC_3.0-100.0GeV.km3sim.jte.jmergefit.offline_81-100.log
│   ├── graphnet_20240515-150618.log
│   ├── graphnet_20240515-150819.log
│   ├── graphnet_20240515-151015.log
│   ├── graphnet_20240515-151230.log
│   ├── graphnet_20240515-151430.log
│   ├── graphnet_20240515-151616.log
│   ├── graphnet_20240515-151811.log
│   ├── graphnet_20240515-152019.log
│   ├── graphnet_20240515-152233.log
│   ├── graphnet_20240515-152443.log
│   ├── graphnet_20240515-152645.log
│   ├── graphnet_20240515-152847.log
│   ├── graphnet_20240515-153254.log
│   ├── graphnet_20240515-153506.log
│   ├── graphnet_20240515-153648.log
│   ├── graphnet_20240515-153842.log
│   ├── graphnet_20240515-154039.log
│   ├── graphnet_20240515-154242.log
│   ├── graphnet_20240515-154452.log
│   ├── graphnet_20240515-154638.log
│   └── SQLite
│       ├── gsg_tau-CC_3.0-100.0GeV.100.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.81.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.82.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.83.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.84.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.85.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.86.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.87.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.88.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.89.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.90.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.91.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.92.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.93.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.94.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.95.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.96.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.97.km3sim.jte.jmergefit.offline.log
│       ├── gsg_tau-CC_3.0-100.0GeV.98.km3sim.jte.jmergefit.offline.log
│       └── gsg_tau-CC_3.0-100.0GeV.99.km3sim.jte.jmergefit.offline.log
└── SQLite
    ├── gsg_tau-CC_3.0-100.0GeV.100.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.81.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.82.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.83.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.84.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.85.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.86.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.87.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.88.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.89.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.90.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.91.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.92.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.93.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.94.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.95.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.96.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.97.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.98.km3sim.jte.jmergefit.offline.db
    ├── gsg_tau-CC_3.0-100.0GeV.99.km3sim.jte.jmergefit.offline.db
    └── gsg_tau-CC_3.0-100.0GeV.km3sim.jte.jmergefit.offline_81-100.DONE
```
As can be seen, graphnet is creating also log files. But those contain the same information as the ones created by `rule data_converter`. Afterwards, in case you'd like to merge the individual files into a single file, you can run:
```
pyhton3 scritps/data_merger.py {input_dir} {output_dir}
```


