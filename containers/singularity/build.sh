#! /bin/bash
home=`realpath "$(dirname "$0")"`
cd $home && sudo singularity build --sandbox nav_gym.sif nav_gym.def