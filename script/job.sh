#!/bin/bash
# ------------------------
# job.sh — HTCondor worker job for CharmingFit
# ------------------------

set -e

echo "=== Starting job on $(date) ==="
echo "Running on host: $(hostname)"
echo "Arguments: $@"

# ------------------------
# Environment
# ------------------------
source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh

# ------------------------
# Working directory (scratch)
# ------------------------
WORKDIR=$(mktemp -d)
cd "$WORKDIR"
echo "Working directory: $WORKDIR"

# ------------------------
# Job parameters
# ------------------------
SEED=$1
OUTFILE=out_${SEED}.root

# ------------------------
# Run your code
# ------------------------
python /afs/cern.ch/user/m/mpierini/work/CharmingPenguin/CharmingFit/run_fit.py \
    --nevents 10000000 \
    --seed ${SEED} \
    --config /afs/cern.ch/work/m/mpierini/CharmingPenguin/CharmingFit/config \
    --output ${OUTFILE}

# ------------------------
# Copy output back to EOS
# ------------------------
EOS_OUT=/eos/user/m/mpierini/CharmingPenguin/CharmingFitOutput
echo "Copying output to EOS..."
xrdcp -f ${OUTFILE} root://eosuser.cern.ch/${EOS_OUT}/${OUTFILE}

# ------------------------
# Cleanup
# ------------------------
cd /
rm -rf "$WORKDIR"

echo "=== Job ${SEED} finished on $(date) ==="
