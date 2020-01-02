# Quick script to calculate turnaround efficiency for 2019 schedules
from spt3g import core
from glob import glob

fpatterns = ['/spt/data/bolodata/fullrate/ra0hdec-44.75/90014997/0*g3',
             '/spt/data/bolodata/fullrate/ra0hdec-52.25/90052695/0*g3',
             '/spt/data/bolodata/fullrate/ra0hdec-59.75/90092716/0*g3',
             '/spt/data/bolodata/fullrate/ra0hdec-67.25/90112930/0*g3']

for fpattern in fpatterns:
    class TimeCounter(object):
        def __init__(self):
            self.time_turnaround = 0
            self.time_constant_velocity = 0

        def __call__(self, frame):
            if frame.type == core.G3FrameType.Scan:
                az = frame['RawBoresightAz']
                delta_t = (az.stop.time - az.start.time) / core.G3Units.second

                if 'Turnaround' in frame.keys() and frame['Turnaround'] is True:
                    self.time_turnaround += delta_t
                else:
                    self.time_constant_velocity += delta_t
    time_counter = TimeCounter()

    fnames = glob(fpattern)
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=fnames)
    #pipe.Add(core.Dump)
    pipe.Add(time_counter)
    pipe.Run()

    print(fpattern)
    print('Time spent in turnarounds = {:.2f}'\
          .format(time_counter.time_turnaround))
    print('Time spent in constant-velocity scans = {:.2f}'\
          .format(time_counter.time_constant_velocity))

    eff = time_counter.time_constant_velocity / \
           (time_counter.time_constant_velocity + time_counter.time_turnaround)
    print('Turnaround efficiency = {:.4f}'.format(eff))
    print()
