#! /bin/bash
# Launch a batched job running a bash script within a container.
# Special arguments used:
# --time=24:00:00 : Run for max of 24 hours (hours, minutes, seconds). To run 2 days: 2-00:00:00
# -A es_hutter : Run on RSL's member share of Euler
# --mail-type=END : Triggers email to $USER@ethz.ch at the end of training
# --warp: run command in ''. Does not work with variables (that's why run.sh is needed).

export run_cmd="python3 /isaac_ws/nav_gym/nav_gym/nav_legged_gym/train/run_train.py "
export custom_flags="--nv -B /cluster/home/$USER/nav_gym/:/isaac_ws/nav_gym"

sbatch \
  -n 16 \
  --mem-per-cpu=2048 \
  -G 1 \
  --gres=gpumem:10240 \
  --time=12:00:00 \
  -A es_hutter \
  --mail-type=END \
  --tmp=15G \
  run.sh 