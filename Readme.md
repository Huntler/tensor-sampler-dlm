# VST Plugin: AI Sampler
## Setup environment
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your system and execute one of the follwing command afterwards, depending on GPU acceleration (CUDA 11.6).

```(1) $ conda env create -f conda-envs/universal.yml```

```(2) $ conda env create -f conda-envs/cuda-11.yml```

After installation, the environment can be activated by calling 

```$ conda activate vst-plugin```

## Train
There are two options to train, the first one trains one model based on a provided configuration file and the second one trains a set of configurations located in a folder. CAUTION: The second method is not implemented yet.

```(1) $ python main.py --config <PATH_TO_CONFIG_FILE>```

```(2) $ python utils/mutli_train.py --folder <PATH_TO_CONFIG_FOLDER>```

## Tensorboard
To see logs of any training, make sure that the conda environment is enabled, then call

```$ tensorboard --logdir=runs```

to show all trainings in tensorboard. Press [here](http://localhost:6006) to access the local webpage.
