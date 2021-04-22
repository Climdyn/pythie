
from core.data import Data
import numpy as np
import datetime


def test_sin(lead_time, step, number_of_member=10, number_of_observation=20,
             bias=1., beta=1., spread_scaling=1., initial_trange=(5, 15), variability=(1, 3)):
    observations = Data()
    reforecasts = Data()

    obs = list()
    rfcs = list()
    time = np.arange(0., lead_time+0.01, step)
    l = len(time)

    for i in range(number_of_observation):
        o = np.random.randint(*initial_trange) \
            + (np.random.randint(*variability) * (np.random.rand() + 1.) * np.cos(2 * np.pi * time / lead_time + np.random.randint(0, 20))) * np.sin(2 * np.pi * (time - 12.) / 24.) \
            + 0.06 * (2 * np.random.rand() - 1) * time
        obs.append(o)
        ensemble_forecasts = Data(o + (0.05 * np.random.rand() + 0.05 * np.random.rand() * time) * np.random.randn(1, 1, number_of_member, 1, l))
        rfcs.append(np.squeeze(bias * (time - 36.) / lead_time
                               + beta * (1.05 - 0.12 * time / lead_time) * ensemble_forecasts.ensemble_mean.data
                               + spread_scaling * (1. + 0.04 * time) * ensemble_forecasts.centered_ensemble.data))

    tt = list()
    for t in time:
        day = int(t // 24) + 1
        h = int(t % 24)
        tt.append(datetime.datetime(year=1900, month=1, day=day, hour=h))
    tt = np.array(tt)

    for o in obs:
        observations.load_scalars(o, load_axis='obs', concat_axis='obs', timestamps=[tt])
    reforecasts.load_scalars(rfcs, load_axis=['obs', 'member'], timestamps=number_of_observation * [tt])

    return observations, reforecasts
