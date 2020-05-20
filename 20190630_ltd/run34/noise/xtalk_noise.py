from pydfmux.spt3g.northern_tuning_script import *
import datetime
import pickle

ob_amps = [0.001, 0.005, 0.008, 0.01, 0.012]
single_mods = ['0135/1/1', '0136/1/1', '0136/2/2']
LCpartner_mods = ['0135/1/4', '0136/1/4', '0136/2/3']
output_filename = '%s_xtalk_noise_hkdata.pkl' % '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
test_configs = ['all_modules'] #'single_low_amp_freq_dependence'] #'1_module', 'LC_pairs', 'all_modules']

low_amplitude_freq_range = [2.5e6, 3.0e6]

hk_data = dict()

run_do_zero_combs()

if '1_module' in test_configs:
    print('Starting single-module noise test...')
    hk_data['1_module'] = dict()
    for amp in ob_amps:
        print('Overbiasing to amplitude = {}'.format(amp))
        hk_data['1_module'][amp] = dict()
        for rmod in single_mods:
            print('Overbiasing module {}'.format(rmod))
            hk_data['1_module'][amp][rmod] = dict()
            bolos = hwm.bolos_from_pstring('{}/*'.format(rmod)).filter(pydfmux.Bolometer.tune == True)
            OB_results = bolos.overbias_and_null_threaded(threads=2,
                                                          carrier_amplitude=amp,
                                                          scale_by_frequency=overbias_scale_by_freq,
                                                          maxnoise=maxnoise,
                                                          shorted_threshold=shorted_threshold,
                                                          max_resistance=max_resistance)
            hk_data['1_module'][amp][rmod]['overbias'] = OB_results['output_directory']

            alive = bolos.find_alive_bolos()
            noise_results = alive.dump_info()
            hk_data['1_module'][amp][rmod]['dump_info'] = noise_results[noise_results.keys()[0]]['output_directory']
            
            run_do_zero_combs()

            with open(output_filename, 'w') as f:
                pickle.dump(hk_data, f)


if 'all_modules' in test_configs:
    print('Starting all module test...')
    hk_data['all_modules'] = dict()
    for amp in ob_amps:
        print('Overbiasing to amplitude = {}'.format(amp))
        hk_data['all_modules'][amp] = dict()
        bolos = hwm.query(pydfmux.Bolometer).filter(pydfmux.Bolometer.tune == True)
        OB_results = bolos.overbias_and_null_threaded(threads=2,
                                                      carrier_amplitude=amp,
                                                      scale_by_frequency=overbias_scale_by_freq,
                                                      maxnoise=maxnoise,
                                                      shorted_threshold=shorted_threshold,
                                                      max_resistance=max_resistance)
        hk_data['all_modules'][amp]['overbias'] = OB_results['output_directory']

        alive = bolos.find_alive_bolos()
        noise_results = alive.dump_info()
        hk_data['all_modules'][amp]['dump_info'] = noise_results[noise_results.keys()[0]]['output_directory']

        run_do_zero_combs()

        with open(output_filename, 'w') as f:
                pickle.dump(hk_data, f)


if 'LC_pairs' in test_configs:
    print('Starting LC pair test...')
    hk_data['LC_pairs'] = dict()
    for amp in ob_amps:
        print('Overbiasing to amplitude = {}'.format(amp))
        hk_data['LC_pairs'][amp] = dict()
        for rmod, partnermod in zip(single_mods, LCpartner_mods):
            print('Overbiasing module {}'.format(rmod))
            hk_data['LC_pairs'][amp][rmod] = dict()
            bolos_rmod = hwm.bolos_from_pstring('{}/*'.format(rmod))
            bolos_partner = hwm.bolos_from_pstring('{}/*'.format(partnermod))
            bolos = bolos_rmod.union(bolos_partner).filter(pydfmux.Bolometer.tune == True)
            OB_results = bolos.overbias_and_null_threaded(threads=2,
                                                          carrier_amplitude=amp,
                                                          scale_by_frequency=overbias_scale_by_freq,
                                                          maxnoise=maxnoise,
                                                          shorted_threshold=shorted_threshold,
                                                          max_resistance=max_resistance)
            hk_data['LC_pairs'][amp][rmod]['overbias'] = OB_results['output_directory']

            alive = bolos.find_alive_bolos()
            noise_results = alive.dump_info()
            hk_data['LC_pairs'][amp][rmod]['dump_info'] = noise_results[noise_results.keys()[0]]['output_directory']
            
            run_do_zero_combs()

            with open(output_filename, 'w') as f:
                pickle.dump(hk_data, f)

                
if 'single_low_amp' in test_configs:
    print('Starting single low-amplitude test...')
    hk_data['single_low_amp'] = dict()

    for amp in ob_amps:
        hk_data['single_low_amp'][amp] = dict()
        for rmod in single_mods:
            print('Overbiasing to amplitude = {}'.format(amp))
            hk_data['single_low_amp'][amp][rmod] = dict()
            
            # overbias all bolometers to desired amplitude
            bolos = hwm.query(pydfmux.Bolometer).filter(pydfmux.Bolometer.tune == True)
            OB_results = bolos.overbias_and_null_threaded(threads=2,
                                                          carrier_amplitude=amp,
                                                          scale_by_frequency=overbias_scale_by_freq,
                                                          maxnoise=maxnoise,
                                                          shorted_threshold=shorted_threshold,
                                                          max_resistance=max_resistance)
            hk_data['single_low_amp'][amp][rmod]['overbias'] = OB_results['output_directory']

            # overbias the module under test to a small amplitude
            bolos = hwm.bolos_from_pstring('{}/*'.format(rmod))
            OB_results = bolos.overbias_and_null_threaded(threads=2,
                                                          carrier_amplitude=0.001,
                                                          scale_by_frequency=overbias_scale_by_freq,
                                                          maxnoise=maxnoise,
                                                          shorted_threshold=shorted_threshold,
                                                          max_resistance=max_resistance)

            # take noise on the low-amplitude module only
            alive = bolos.find_alive_bolos()
            noise_results = alive.dump_info()
            hk_data['single_low_amp'][amp][rmod]['dump_info'] = noise_results[noise_results.keys()[0]]['output_directory']

            run_do_zero_combs()

            with open(output_filename, 'w') as f:
                pickle.dump(hk_data, f)

if 'single_low_amp_freq_dependence' in test_configs:
    print('Starting single low-amplitude test with frequency-dependence...')
    hk_data['single_low_amp_freq_dependence'] = dict()

    for amp in ob_amps:
        hk_data['single_low_amp_freq_dependence'][amp] = dict()
        for rmod in single_mods:
            print('Overbiasing to amplitude = {}'.format(amp))
            hk_data['single_low_amp_freq_dependence'][amp][rmod] = dict()
            
            # overbias all bolometers to desired amplitude
            bolos = hwm.query(pydfmux.Bolometer).filter(pydfmux.Bolometer.tune == True)
            OB_results = bolos.overbias_and_null_threaded(threads=2,
                                                          carrier_amplitude=amp,
                                                          scale_by_frequency=overbias_scale_by_freq,
                                                          maxnoise=maxnoise,
                                                          shorted_threshold=shorted_threshold,
                                                          max_resistance=max_resistance)
            hk_data['single_low_amp_freq_dependence'][amp][rmod]['overbias'] = OB_results['output_directory']

            # lower all carriers the desired frequency range
            bolos_freq_range = bolos.join(pydfmux.ChannelMapping, pydfmux.ReadoutChannel, pydfmux.LCChannel)\
                                    .filter((pydfmux.LCChannel.frequency > low_amplitude_freq_range[0]) & \
                                            (pydfmux.LCChannel.frequency < low_amplitude_freq_range[1]))
            OB_results = bolos_freq_range.overbias_and_null_threaded(threads=2,
                                                                     carrier_amplitude=0.001,
                                                                     scale_by_frequency=overbias_scale_by_freq,
                                                                     maxnoise=maxnoise,
                                                                     shorted_threshold=shorted_threshold,
                                                                     max_resistance=max_resistance)
            print(bolos.count())
            print(bolos_freq_range.count())
            
            bolos = hwm.bolos_from_pstring('{}/*'.format(rmod))            
            OB_results = bolos.overbias_and_null_threaded(threads=2,
                                                          carrier_amplitude=0.001,
                                                          scale_by_frequency=overbias_scale_by_freq,
                                                          maxnoise=maxnoise,
                                                          shorted_threshold=shorted_threshold,
                                                          max_resistance=max_resistance)

            # take noise on the low-amplitude module only
            alive = bolos.find_alive_bolos()
            noise_results = alive.dump_info()
            hk_data['single_low_amp_freq_dependence'][amp][rmod]['dump_info'] = noise_results[noise_results.keys()[0]]['output_directory']

            run_do_zero_combs()

            with open(output_filename, 'w') as f:
                pickle.dump(hk_data, f)
