#!/bin/sh

echo Executing on host: `hostname`
output=$1
shift
echo Output file: $output
echo Input files: $@

toolset=py3-v3

# Set SHELL so setup.sh knows to do the right thing (it gets set to tcsh sometimes)
export SHELL=sh

export SCRIPT=master_field_mapmaker_signflip.py

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

# Get input files
echo 'Transferring input files to scratch directory' $_CONDOR_SCRATCH_DIR
INPUT_URLS=""
LOCAL_PATH=""
for file in $@; do
        if [ `echo $file | head -c 1` != '/' ]; then
                # actually some kind of argument
                LOCAL_PATH="$LOCAL_PATH $file"
        else
            INPUT_URLS="$INPUT_URLS gsiftp://osg-gridftp.grid.uchicago.edu:2811$file"
            LOCAL_PATH="$LOCAL_PATH data/`basename $file`"
        fi
done
mkdir data
for url in $INPUT_URLS; do
        echo Acquiring $url...
	gfal-copy -t 7200 $url file://$_CONDOR_SCRATCH_DIR/data/
	#if [[ "$FS" == "DCACHE" ]]; then
	#    gfal-copy $url file://$_CONDOR_SCRATCH_DIR/data/
	#elif [[ "$FS" == "ceph" ]]; then
        #    globus-url-copy -rst $url file://$_CONDOR_SCRATCH_DIR/data/
	#fi
done

# Download software
mkdir software
software_url=gsiftp://osg-gridftp.grid.uchicago.edu:2811/sptgrid/user/adama/tarballs/20201007_spt3g_py3-v3_RHEL_7_x86_64.tgz
echo 'Downloading software distribution' $software_url
gfal-copy $software_url software/spt3g_software.tgz
cd software
tar xzf spt3g_software.tgz
cd ..

# Run processing
export PYTHONPATH=`pwd`/software/:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/software/spt3g:$LD_LIBRARY_PATH
export SPT3G_SOFTWARE_BUILD_PATH=`pwd`/software
export OMP_NUM_THREADS=1
echo python $SCRIPT -o ./`basename $output` $LOCAL_PATH
python $SCRIPT -o ./`basename $output` $LOCAL_PATH

# Send back output
gfal-copy file://$PWD/`basename $output` gsiftp://osg-gridftp.grid.uchicago.edu:2811$output

trap - EXIT

echo Finished
