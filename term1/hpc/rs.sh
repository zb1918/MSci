#!/bin/sh

#PBS -N job
#PBS -J 1-10
#PBS -l walltime=03:00:00

#PBS -l select=1:ncpus=8:mem=90000mb:mpiprocs=1:ompthreads=8

module load anaconda3/personal

OUTDIR=$WORK/optical

rm -r $OUTDIR
mkdir -p $OUTDIR

cp -r $HOME/MSci/* $OUTDIR/

cd $OUTDIR

python $HOME/MSci/9_optical_depth.py $PBS_ARRAY_INDEX

