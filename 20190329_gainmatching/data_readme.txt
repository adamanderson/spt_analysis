--Horizon noise stare (77863968), for Amy's LTD proceeding, poly 1 filter, nuclear gain matching, calculate pair sum and pair difference ASDs. Incantation:
python test_gain_match_and_fit.py /spt/data/bolodata/downsampled/noise/73798315/offline_calibration.g3 /spt/data/bolodata/fullrate/noise/77863968/0000.g3 -o horizon_noise_77863968_bender_ltd.g3 --average-asd --fit-asd --units current --poly-order 1 --fit-readout-model --gain-match --diff-pairs --sum-pairs --group-by-band --group-by-wafer

--Same as above, but only the per bolo ASDs and fits. No gain matching or pair sum/difference. Incantation:
python test_gain_match_and_fit.py /spt/data/bolodata/downsampled/noise/73798315/offline_calibration.g3 /spt/data/bolodata/fullrate/noise/77863968/0000.g3 -o horizon_noise_77863968_bender_ltd_perbolo_only.g3 --fit-asd --units current --poly-order 1 --fit-readout-model --per-bolo-asd
