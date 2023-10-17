#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

export CARB_APP_PATH=$ISAAC_HOME/kit
export EXP_PATH=$ISAAC_HOME/apps
export ISAAC_PATH=$ISAAC_HOME
source $SCRIPT_DIR/setup_python_env.sh
