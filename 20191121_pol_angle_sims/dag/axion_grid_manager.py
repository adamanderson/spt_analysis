from spt3g import cluster, core
import argparse
import yaml
import os
import textwrap

parser = argparse.ArgumentParser('Tools for grid submission of simulations related '
                                 'to the axion oscillation analysis.',
                                 formatter_class=argparse.RawTextHelpFormatter)
cli = parser.add_argument_group('Command Line Inputs', 'These settings are '
                                'specified via the command line only:')
cli.add_argument('--config-file',
                 help = ('.yaml file containing map-making parameters. '
                         'These will override command-line settings '
                         'for the options below.'))
cli.add_argument('--actions', choices=['noise', 'cmb', 'mock_observe'],
                 help=textwrap.dedent('''\
                     Action to take:
                         `noise` : noise-only simulations
                         `cmb` : CMB-only simulations
                         `mock_observe` : mock-observe simulated maps
                      '''))
config = parser.add_argument_group('Config File Inputs', 'These settings '
                                   'are specified in the config file:')
config.add_argument('--sim-script',
                    default='{}/../simulations/scripts/make_3g_sims.py'\
                    .format(os.environ['SPT3G_BUILD_ROOT']),
                    help='Path to script that constructs simulations.')
config.add_argument('--mock-script',
                    default='{}/../std_processing/std_processing/mapmakers/master_field_mapmaker.py'\
                    .format(os.environ['SPT3G_BUILD_ROOT']),
                    help='Path to script that does mock-observing.')
config.add_argument('--condor-dir',
                    help='Directory where condor output files are stored.')
config.add_argument('--requirements',
                    default="""( ( HAS_CVMFS_spt_opensciencegrid_org ) && '
                    '( ( ( TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName1 ) || ( RCC_Factory == "ciconnect" ) ) && '
                      '( ( TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName2 ) || ( RCC_Factory == "ciconnect" ) ) && '
                      '( ( TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName3 ) || ( RCC_Factory == "ciconnect" ) ) && '
                      '( ( TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName4 ) || ( RCC_Factory == "ciconnect" ) ) && '
                      '( ( TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName5 ) || ( RCC_Factory == "ciconnect" ) ) ) && '
                    '( OSGVO_OS_STRING == "RHEL 7" ) && (GLIDEIN_ResourceName =!= "NPX") && '
                    '( GLIDEIN_SITE =!= "SU-ITS") && (GLIDEIN_SITE =!= "UColorado_HEP"))""",
                    help='Requirements for condor job.')
config.add_argument('--request-memory',
                    default=2 * core.G3Units.GB,
                    help='Memory to request per slot.')
config.add_argument('--request-disk',
                     default=1 * core.G3Units.GB,
                     help='Disk space to request per slot.')
config.add_argument('--grid-proxy',
                    default='/home/ddutcher/ddutcher_proxy',
                    help='Grid proxy.')
args = parser.parse_args()

# variables that do not change
globus_uri = 'gsiftp://ceph-gridftp1.grid.uchicago.edu:2811/cephfs'

# If configuration yaml is specified, load it and pull parameters from there.                                                       
if args.config_file is not None:
    settings = yaml.safe_load(open(args.config_file, 'r'))
    for k, v in settings.items():
        setattr(args, k, v)

def build_dag_single_obs(actions,
                         jobname,
                         sim_stub,
                         sim_dir,
                         condor_dir,
                         sim_config=None,
                         sim_script=args.sim_script,
                         mock_script=args.mock_script,
                         sim_index=1,
                         num_sims=1,
                         pol_sims=True):
    '''
    Build a DAG file for simulations and/or mock-observing for the axion
    analysis.

    Parameters
    ----------
    actions : list
        List of actions that are supposed to be performed by the DAG:
            'noise' : construct noise sims
            'cmb' : construct simulation with cmb
            'mock_observe' : do mock observations
    jobname : str
        Name of job to run
    sim_stub : str
        Name of file with sim stub
    sim_dir : str
        Directory where simulation output should go
    sim_config : str
        Name of simulation config file
    sim_script : str
        Name of script to handle simulations
    mock_script : str
        Name of script to handle mock observations
    sim_index : int
        Simulation index
    num_sims : int
        Number of simulations to run
    pol_sims : bool
        Do polarized simulation

    Returns
    -------
    None
    '''
    if 'noise' or 'cmb' in actions:
        sim_args = ['-c', sim_config,
                    '-o', sim_dir,
                    '--sim-index', sim_index,
                    '--num-sims', num_sims]

        if 'noise' in actions:
            sim_args.append('--noise')
        if 'cmb' in actions:
            sim_args.append('--cmb')
        if pol_sims:
            sim_args.append('--pol')
            
        cluster.condor_submit(
                sim_script,
                create_only=True,
                args=sim_args,
                log_root=condor_dir,
                output_root=sim_dir,
                verbose=False,
                retry=False,
                jobname=jobname,
                input_files=[simstub],
                user_code="",
                aux_input_files=[sim_config],
                output_files=output_files, ##
                requirements=requirements,
                request_disk=args.request_disk,
                request_memory=args.request_memory,
                grid_proxy=args.grid_proxy,
                globus_uri=globus_uri,
            )


if __name__ == '__main__':
    build_dag_single_obs(actions,
                         jobname,
                         sim_stub,
                         sim_dir,
                         condor_dir)
