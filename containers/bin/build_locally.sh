#!/bin/bash

# This script can be used to debug the container locally
# It will mount the current directory into the container
# and run a shell inside the container. This allows you to
# run the container locally and debug it inside. (thanks Github Copilot)
ISAAC_WS="${PWD}/../../../"

# custom_flags="--nv --writable -B $ISAAC_WS/nav_gym:/isaac_ws/nav_gym"
custom_flags="--nv -B $ISAAC_WS/nav_gym:/isaac_ws/nav_gym"
echo $custom_flags

singularity shell $custom_flags $ISAAC_WS/nav_gym/containers/singularity/nav_gym.sif
