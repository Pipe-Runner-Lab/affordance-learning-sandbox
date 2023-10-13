# Project Mirage

- [Project Mirage](#project-mirage)
- [🧪 Setup](#-setup)
  - [🦾 Nvidia Omniverse](#-nvidia-omniverse)
    - [⚛️ Nucleus](#️-nucleus)
  - [🌿 Conda](#-conda)
    - [🐍 Python](#-python)
    - [📦 Dependencies](#-dependencies)
  - [📂 VS Code](#-vs-code)
  - [🔼 OmniIsaacGymEnvs](#-omniisaacgymenvs)
- [⏯️ Starting the Simulation](#️-starting-the-simulation)
- [📝 FAQ](#-faq)

# 🧪 Setup

The following steps are required to run the project.

## 🦾 Nvidia Omniverse

The project uses Nvidia Omniverse to render the simulation. The Omniverse application can be downloaded from the [Nvidia Omniverse website](https://www.nvidia.com/en-us/design-visualization/omniverse/).

Once installation is done, add the `ISAAC_HOME` path to the environment variable list. This points to the root of Isaac sim installation. For example, my installation is at `D:\Tools\Omniverse\pkg\isaac_sim-2022.2.1`. This is how I have added it to my environment variables on Windows:

<center>
<img src="./docs/images/env_vars.png" alt="Environment Variables" width="500"/>
</center>

**Note:** This is important for the environment setup script uses this variable to complete the rest of the setup.

### ⚛️ Nucleus

The USD files for Franka robots and a few other models are used from the Nvidia Omniverse Nucleus server. The Nucleus server can be started by running the following command.
Verify that the Nucleus server is running by navigating to http://localhost:3080 in a web browser.

## 🌿 Conda

The project uses conda to manage the python environment. The conda package manager can be installed by following the instructions on the [conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

### 🐍 Python

We start off by creating and activating a new environment for the project. We will use python version 3.7.

```bash
conda remove --name isaac-sim --all # Remove the environment if it already exists
conda env create -f environment.yml
conda activate isaac-sim
```

### 📦 Dependencies

The project dependencies can be installed by running the following command from the root of the project.

**Note:** Ensure that pytorch is able to detect the GPU else the training scripts won't work. I had to switch to a different version of cuda for pytorch to detect the GPU. This is how I did it on Windows:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 📂 VS Code

`.vscode/settings.json` contains the settings for VS Code. The settings file is used to setup the python interpreter and additional packages provided by isaac sim for intellisense. There are a few caveats right now on Windows for the environment variables to be picked up by VS Code. I have hardcoded the value to the value of `ISAAC_HOME` for now.

## 🔼 OmniIsaacGymEnvs

The project uses the OmniIsaacGymEnvs package to train RL algorithms using Isaac sim as an environment. The package can be installed by following the instructions on the [OmniIsaacGymEnvs GitHub page](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs).

# ⏯️ Starting the Simulation

The project has to be run from the root of the project.

```bash
python -m src.mirage task=""
tensorboard --logdir runs
```

# 📝 FAQ

- Q. Cannot import `omni.isaac.core`?
  - A. https://forums.developer.nvidia.com/t/cannot-import-omni-isaac-core/242977/2
- Q. How to setup VS Code for Omniverse?
  - A. https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html#isaac-sim-python-vscode
