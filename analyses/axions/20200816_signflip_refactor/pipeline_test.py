from spt3g import core

pipe = core.G3Pipeline()

pipe.Add(core.G3Reader, filename='/sptgrid/user/kferguson/axion_initial_2019_maprun/simstub_ra0hdec-52.25_150GHz_73756603.g3.gz')
pipe.Add(core.G3Writer, filename='test.g3')
pipe.Run()
