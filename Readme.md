# Nav gym

# HPC workflow

## 3. Move container to Cluster
First, you need to pass the container to the cluster. You can use either scp or rsync. First compress using tar into a single file.

### 3.1 On Cluster

```bash

mkdir -p /cluster/work/rsl/$USER/logs
echo "export WORK=/cluster/work/rsl/$USER" >> ~/.bashrc

```

### 3.2 On your PC:

```bash

cd ~/isaac_ws/nav_gym/containers/singularity
sudo tar -cvf navgym.tar navgym.sif
sudo scp -v navgym.tar $EULER_USER@euler.ethz.ch:/cluster/work/rsl/$EULER_USER

```

### *3.3:On the cluster (only necessary for debugging the singularity container on the cluster, otherwise, this step can be skipped):

```bash

tar -xf $WORK/navgym.tar -C $SCRATCH

```

## 4.Running your training script inside container
Create and clone git folders (do not try to install them):

```bash

cd $HOME
mkdir git && cd git
git clone https://github.com/yuxuanxienova/nav_gym.git

```

## 5.Running on GPU



