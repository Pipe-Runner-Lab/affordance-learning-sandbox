# Project Mirage

# 🧪 Setup
The following steps are required to run the project.

## 🦾 Nvidia Omniverse
The project uses Nvidia Omniverse to render the simulation. The Omniverse application can be downloaded from the [Nvidia Omniverse website](https://www.nvidia.com/en-us/design-visualization/omniverse/).

### ⚛️ Nucleus
The USD files for Franka robots and a few other models are used from the Nvidia Omniverse Nucleus server. The Nucleus server can be started by running the following command.
Verify that the Nucleus server is running by navigating to http://localhost:3080 in a web browser.

## 🌿 Conda
The project uses conda to manage the python environment. The conda package manager can be installed by following the instructions on the [conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

## 🐍 Python
We start off by creating and activating a new environment for the project. We will use python version 3.7.
```bash
conda create -n mirage_env python=3.7
conda activate mirage_env
```

## 📦 Dependencies
The project dependencies can be installed by running the following command from the root of the project.
```bash
pip install -r requirements.txt
```

## 📂 Adding Omniverse Dependencies
`OMNIVERSE_PATH`
TBD

## 🔼 OmniIsaacGymEnvs
The project uses the OmniIsaacGymEnvs package to train RL algorithms using Isaac sim as an environment. The package can be installed by following the instructions on the [OmniIsaacGymEnvs GitHub page](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs).

# ⏯️ Starting the Simulation
The project has to be run from the root of the project.
```bash
python ./src/main.py
```