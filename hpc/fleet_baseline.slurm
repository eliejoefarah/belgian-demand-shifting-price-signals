#!/bin/bash
# Phase 2 fleet baseline on CÉCI Lyra (ULB), SLURM job array. See temp_phase2_log.md section 5.
#
# Lyra (checked 8 Nov 2026): partition batch (default, 5 days max) and debug (1 h, 16 cores
# per user); 256 cores per user (QOS normal); nodes have 32 cores and 132 GB; Rocky Linux 9.8;
# no Gurobi/EnergyPlus/Python 3.13 modules, so our own ~/miniforge3 is used.
#
# Submit from the repo root. SLURM opens --output before the script runs, so logs/ must
# exist first. --export=ALL keeps PATH/HOME etc.; without ALL only the listed variables go.
#   mkdir -p logs
#   sbatch --export=ALL,YEAR=2023 hpc/fleet_baseline.slurm
#   sbatch --export=ALL,YEAR=2022 hpc/fleet_baseline.slurm
# Smoke test first (1 week, 2 houses per stratum, one task on the debug partition):
#   sbatch --array=0 --partition=debug --cpus-per-task=16 --time=01:00:00 \
#       --export=ALL,YEAR=2023,EXTRA="--houses smoke --days 7 --name smoke_2023 --n-shards 1" hpc/fleet_baseline.slurm
# (EXTRA's --n-shards 1 comes after the script's --n-shards 16; argparse keeps the last one.)
# Resubmit only some failed tasks: --array=3,7 (the script always passes --n-shards 16, so
# task k still takes houses k, k+16, ...; keep it equal to the size of --array=0-15).
#
# Resumable: resubmitting the same command only runs houses without a house_{n}.json, and
# stops if a done house was saved with other settings (e.g. a smoke test in the same folder).
#
# Sizing: 16 tasks x 16 cores = 256 cores = the per-user limit. Half-node tasks, because the
# batch nodes are usually partly busy (8 Nov: ~21 idle cores per mixed node), so 32-core
# tasks would wait for a fully free node. 500 houses / 16 = 32 houses per task -> 2 rounds of
# 16. Laptop timings: ~30 min per house-year alone, ~65 min per worker under load, so
# ~1-2.2 h per task; --time has a margin. Replace with the smoke-test timing.
# Memory: ~0.6 GB per worker measured, Gurobi SoftMemLimit allows 2 GB per solve:
# 3 GB per core = 48 GB per task (nodes have 132 GB).

#SBATCH --job-name=fleet_baseline
#SBATCH --partition=batch
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=06:00:00
#SBATCH --output=logs/fleet_%A_%a.out

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
set +u  # some conda activation scripts read unset variables
conda activate belgian_dwellings
set -u

# Gurobi academic WLS licence 2882419 (expires 7 Feb 2027, extendable; WLS needs internet on
# the node). The file holds a secret: chmod 600, never commit it.
export GRB_LICENSE_FILE=$HOME/gurobi.lic

# EnergyPlus temp run folders (tempfile) on the node-local disk, not the shared filesystem.
export TMPDIR="${LOCALSCRATCH:-${TMPDIR:-/tmp}}"

# One Gurobi thread per house, and keep numpy/BLAS from spawning extra threads per worker.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-}  cpus ${SLURM_CPUS_PER_TASK}  TMPDIR $TMPDIR"
python -u -m experiments.fleet_baseline --year "${YEAR:?set YEAR=2022 or 2023}" \
    --workers "${SLURM_CPUS_PER_TASK}" --n-shards 16 ${EXTRA:-}
