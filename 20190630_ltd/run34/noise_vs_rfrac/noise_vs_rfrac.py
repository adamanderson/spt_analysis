from pydfmux.spt3g.northern_tuning_script import *
import datetime
import pickle

output_filename = '%s_noise_v_rfrac_hkdata.pkl' % '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
rfracs = [0.95, 0.9, 0.85, 0.8, 0.75]
rmod = '0135/1/*'
bolos = hwm.bolos_from_pstring('{}/*'.format(rmod))
overbias_amp = 0.012
hk_data = dict()

# preemptively overbias
bolos_alive = bolos.find_alive_bolos().filter(pydfmux.Bolometer.tune == True)
overbias_results = bolos_alive.overbias_and_null(carrier_amplitude=0.013,
                                                 scale_by_frequency=True)
alive = bolos.find_alive_bolos()
noise_results = alive.dump_info()
hk_data[1.0] = noise_results[noise_results.keys()[0]]['output_directory']

for rfrac in rfracs:    
    for bolo in bolos:
        bolo.rfrac = rfrac
    
    # check bolometer states and only drop overbiased bolos
    for bolo in bolos:
        if bolo.readout_channel:
            bolo.state = bolo.retrieve_bolo_state().state
        hwm.commit()
    bolos_to_drop = bolos.filter(pydfmux.Bolometer.state=='overbiased')
    drop_bolos_results = bolos_to_drop.drop_bolos(A_STEP_SIZE=0.00001,
                                                  fixed_stepsize=False,
                                                  TOLERANCE=0.02)

    alive = bolos.find_alive_bolos()
    noise_results = alive.dump_info()
    hk_data[rfrac] = noise_results[noise_results.keys()[0]]['output_directory']

    # check bolometer states and only drop tuned bolos
    for bolo in bolos:
        if bolo.readout_channel:
            bolo.state = bolo.retrieve_bolo_state().state
        hwm.commit()
    bolos_to_overbias = bolos.filter(pydfmux.Bolometer.state=='tuned')
    overbias_results = bolos_to_overbias.overbias_and_null(carrier_amplitude=overbias_amp,
                                                           scale_by_frequency=True)
    
    with open(output_filename, 'wb') as f:
        pickle.dump(hk_data, f)
    
