#! /bin/bash
home=`realpath "$(dirname "$0")"/../../../`
cd $home && sudo docker build --network=host --memory-swap -1 -m 20g -t navgym -f nav_gym/containers/docker/Dockerfile .
