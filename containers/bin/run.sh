#! /bin/bash
function print () { echo "$(date): $@"; }

print "Started untaring to TMPDIR"
tar -xf $WORK/navgym.tar -C ${TMPDIR}
print "Sucessfully untared to TMPDIR. Starting singularity exec."
singularity exec ${custom_flags} ${TMPDIR}/navgym.sif bash -c "${run_cmd}"
