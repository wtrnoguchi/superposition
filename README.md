# Superposition mechanism as a neural basis for understanding others

This repository provides an implementation of the simulation experiments in the paper "Superposition mechanism as a neural basis for understanding others" by Wataru Noguchi, Hiroyuki Iizuka, Masahito Yamamoto, and Shigeru Taguchi.

Paper information will be updated after acceptance.

All the datasets and results of the paper can be reproduced by the codes in this repository.

## Dependencies

To reproduce the results, please setup the following dependencies.
Or container environment of singularity can be used ([see below](#using-singularity-container)).

### System dependencies

The codes were tested on the system with ...

- Ubuntu 20.04
- Python 3.8
- CUDA 11.4

Although not tested, the code can be run on systems with different versions of software.

### Other ubuntu package dependencies

Install using apt.

```
$ sudo apt install imagemagick libopencv-dev
```

### Python package dependeicies

Python package dependencies can be installed by using pip.

```
$ pip install -r requirements.txt
```

## Run all experiments

All the results on the paper can be reproduced by running following script.

```
$ ./run_all.sh
```

The results of analyses can be found the locations listed on [docs/results_locations.md](docs/results_locations.md).

The above script excecute following scripts for the data collection, trainings, and analyses.

## Collecting dataset for training and analysis

First, collect datasets used for the training of the network.

```
$ ./run_collect_data.sh
```

## Run training

Then, perform all the trainings explained in the paper.

```
$ ./run_training.sh
```

The training logs and trained network parameters will be saved under the directory `data/result`

## Run analyzing and visualizing training results

Finally, perform the analysis on the trained networks.

```
$ ./run_analysis.sh exp1
```

```
$ ./run_analysis_exp2.sh
```

```
$ ./run_analysis.sh exp3
```

```
$ ./run_regression.sh
```

## Using Singularity container

We provide a definition file for singularity container environment (https://sylabs.io/singularity).
CUDA still need to be installed outside the container.

### Creating container environment:

1. Install singularity following the documentation (https://sylabs.io/docs).
1. Build the container.
   ```
   $ singularity build --fakeroot --sandbox env/ singularity.def
   ```

### Running codes in the singularity container:

1. Enter the container.
   ```
   $ singularity shell --fakeroot --nv env
   ```
1. Then, run commands.
   ```
   Singularity> ./run_all.sh
   ```
