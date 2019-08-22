#!/usr/bin/env bash

#process one or several regions

#parameters
region_dir=/data/lidar2/RGI_paper/worldwide/03_rgi60/aster_l1a/
out_dir=/data/lidar2/RGI_paper/worldwide_out/03_rgi60/aster_dem/
proc_dir=/tmp/
log_dir=/home/hugonnet/code/MMASTER-workflows-chris_updates/

#wrapper
cd ${region_dir}

utm_dirs=$(ls -d ${region_dir}*)

cd ${log_dir}
echo "Queueing: sbatch_utm_mmaster.sh ${out_dir} ${proc_dir} ${utm_dirs}"
sbatch sbatch_utm_mmaster.sh ${out_dir} ${proc_dir} ${utm_dirs} &

echo 'Fin.'

