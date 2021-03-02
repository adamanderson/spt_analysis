#!/bin/sh

echo Executing on host: `hostname`

toolset=py3-v3

allargs=( $@ )
len=${#allargs[@]}
output=${allargs[$len-1]}
args=${allargs[@]:0:$len-1}

# Set SHELL so setup.sh knows to do the right thing (it gets set to tcsh sometimes)
export SHELL=sh

export SCRIPT=simulate_time_domain.py

# Set up proxy if needed
export connectHTTPProxy=${connectHTTPProxy:-UNAVAILABLE}
if [ "$connectHTTPProxy" != UNAVAILABLE ]; then
  export http_proxy=$connectHTTPProxy
fi
export OSG_SQUID_LOCATION=${OSG_SQUID_LOCATION:-UNAVAILABLE}
if [ "$OSG_SQUID_LOCATION" != UNAVAILABLE ]; then
  export http_proxy=$OSG_SQUID_LOCATION
fi

echo OSG proxy:  $OSG_SQUID_LOCATION
echo Using proxy:  $http_proxy

env_on_error() {
        echo 'Environment at error:'
        uname -a
        printenv
}

trap env_on_error EXIT

set -e

eval `/cvmfs/spt.opensciencegrid.org/$toolset/setup.sh`

# Stats
echo 'Operating System: ' `uname`
echo 'CPU Architecture: ' `uname -p`
echo 'Platform: ' $OS_ARCH

# Move to scratch dir, copying files as necessary if that is not already where we are
[ -e $_CONDOR_SCRATCH_DIR/$SCRIPT ] || cp $SCRIPT $_CONDOR_SCRATCH_DIR/
cd $_CONDOR_SCRATCH_DIR

# Run script
echo python $SCRIPT $args ./`basename $output`
python $SCRIPT $args ./`basename $output`

# Send back output
gfal-copy file://$PWD/`basename $output` gsiftp://osg-gridftp.grid.uchicago.edu:2811$output

# trap - EXIT

echo Finished

exit 0