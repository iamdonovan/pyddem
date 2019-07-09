#!/usr/bin/env bash
#SBATCH -p defq # partition (queue)
#SBATCH --nodes=16
#SBATCH --cpus-per-task 4
#SBATCH --time=240:00:00
##SBATCH -w, --nodelist=compute1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=romain.hugonnet@gmail.com

#@author: hugonnet
#SLURM wrapper for utm zone processing of MicMac/MMASTER (Luc Girod, Chris Nuth, Robert McNabb)
#should be easily adaptable to qsub & others workload managers for HPCs

#basic usage: "sbatch sbatch_mmaster.sh tdir procdir datadir1 datadir2...

#environment variables
source $HOME/.bashrc
export PATH=/home/hugonnet/code/MMASTER-workflows-chris_updates:$PATH

#tdir: path to target directory to write final results: usually on storage disks
tdir=$1
#procdir: path to directory where processing is done: usually on a HPC node or a SSD
procdir=$2

#all other arguments passed are considered as datadirs: parent directories where l1a strips are located
# (utm zone directory): usually on storage disks
shift 2


#number of tasks at a time: we find that the best value of "CPU per MicMac task" is around 4.
#here we have enough disk speed and RAM to use 4, but depending on the hardware, using more to avoid disk throttling
# can yield better speed
#it really gets inefficient over 12 CPUs per task
nr_task=144 #for 576 computing cores

#cdir=$(pwd)
echo Started `date`

#xargs is one of the best option to run serial independent processes in parallel
#some useful doc for SLURM: https://help.rc.ufl.edu/doc/Sample_SLURM_Scripts
#and here too: https://scitas-data.epfl.ch/kb/Running+multiple+tasks+on+one+node
for datadir in "$@"; do
    cd ${datadir}
#    ls -d AST*/ >> ${cdir}/tmp_l1a.txt
    ls -d AST*/ | sed 's/.$//' | xargs -I{} --max-procs=${nr_task} bash -c "srun -N 1 -n 1 --exclusive process_l1a.sh $datadir/{} $tdir $procdir"

done

#cat tmp_l1a.txt | head -1 | sed 's/.$//' | xargs -I{} --max-procs=${nr_task} bash -c "srun -N 1 -n 1 --exclusive process_l1a_mmaster.sh $datadir/{} $tdir $procdir"
#cat tmp_l1a.txt | head -36 | sed 's/.$//' | xargs -I{} --max-procs=$nr_task bash -c "srun -N 1 -n 1 --exclusive echo $datadir/{} $tdir $procpdir"
wait

echo Finished `date`


