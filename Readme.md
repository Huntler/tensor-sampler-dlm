# VST Plugin: AI Sampler
## Setup environment
### Development Tools
Install the following programs:

- [Visual Studio Code (VS Code)](https://code.visualstudio.com/download)
- [git](https://git-scm.com/downloads)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Initial Setup
Open VS Code go to "Terminal -> New Terminal" and type

```git clone https://github.com/Huntler/tensor-sampler-dlm```

to download the source code. The open the downloaded repository "tensor-sampler-dlm" (Ctrl+K Ctrl+O). Then install relevant dependencies using Miniconda. To do so, open the terminal within VS Code and type the first command if no GPU acceleration is available, otherwise type the second command.

```(1) $ conda env create -f conda-envs/universal.yml```

```(2) $ conda env create -f conda-envs/cuda-11.yml```

### VS Code Plugins
Some plugins are highly recommended, such as the "python" plugin. Install the following list of plugins for the best experience:

- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- audio-preview (sukumo28.wav-preview)
- GitLens (eamodio.gitlens)
- Resource Monitor (mutantdino.resourcemonitor)
- Todo Tree (Gruntfuggly.todo-tree)

### Finishing Setup
Open the "main.py" file and click on the Python's version number on the bottom-right corner. Now, select the previous installed conda environment "vst-plugin" in the opened pop-up. Restart VS Code and open "tensor-sampler-dlm" again.

Happy Coding!

### Additional Settings
To simplify the program start or to enable debugging, create the directory ".vscode" in the project's root. Then, create a file called "launch.json" and paste the following:
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Multi-Train: config",
            "type": "python",
            "request": "launch",
            "program": "utils/multi_train.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                "--folder",
                "config"
            ]
        },
        {
            "name": "Train: Default.yml",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                "--config",
                "config/default.yml"
            ]
        }
    ]
}
```

To start debugging, press "F5". The debug configuration can be changed under the 4th icon on the left-hand menu.

## Train
There are two options to train, the first one trains one model based on a provided configuration file and the second one trains a set of configurations located in a folder. CAUTION: The second method is not implemented yet. IMPORTANT: The datasat should be located at "data/dataset/[dataset_name]/[input.mid|output.wav]".

```(1) $ python main.py --config <PATH_TO_CONFIG_FILE>```

```(2) $ python utils/multi_train.py --folder <PATH_TO_CONFIG_FOLDER>```

## Tensorboard
To see logs of any training, make sure that the conda environment is enabled, then call

```$ tensorboard --logdir=runs```

to show all trainings in tensorboard. Press [here](http://localhost:6006) to access the local webpage.
