#! /bin/bash
function print () { echo "$(date): $@"; }

print "Started untaring to TMPDIR"
tar -xf $WORK/navgym.tar -C ${TMPDIR}
print "Sucessfully untared to TMPDIR. Starting singularity exec."
# Create writable cache directories inside TMPDIR
# mkdir -p "${TMPDIR}/torch_extensions" "${TMPDIR}/cache"
# print "Starting Singularity exec with updated run_cmd."
singularity exec ${custom_flags} ${TMPDIR}/navgym.sif bash -c "${run_cmd}"
