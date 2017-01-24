# coding: utf-8

from app.util import *
from app.som.modsom import SOM
from collections import namedtuple
from app import R, T, L
import matplotlib.pyplot as plt
from app.util.inputmaker import _sample_xlsx
import numpy as np

if __name__ == '__main__':
    sample_cnt = 1

    Xl = namedtuple('Xl', 'path, sheets, col, label')
    xls = [Xl(R('data/raw/invectest/jump.xlsx'), ['Sheet'], 'A', 'jump'),
           Xl(R('data/raw/invectest/run.xlsx'), ['Sheet6', 'Sheet5', 'Sheet4'], 'F', 'run'),
           Xl(R('data/raw/invectest/walk.xlsx'), ['Sheet4', 'Sheet1'], 'F', 'walk')]

    fc = {'jump': [1, 0, 0],
          'run': [0, 1, 0],
          'walk': [0, 0, 1]}
 
    """
    xls = [Xl(R('data/acc/pass_acc_128p_131data.xlsx'), ['Sheet1'], 'A', 'pass'),
           Xl(R('data/acc/placekick_acc_128p_101data.xlsx'), ['Sheet1'], 'A', 'pkick'),
           Xl(R('data/acc/run_acc_128p_132data.xlsx'), ['Sheet1'], 'A', 'run'),
           Xl(R('data/acc/tackle_acc_128p_111data.xlsx'), ['Sheet1'], 'A', 'tackle'),
           Xl(R('data/raw/invectest/walk.xlsx'), ['Sheet4', 'Sheet1'], 'F', 'walk')]
    
    fc = {'pass': [1, 0, 1],
          'pkick': [0, 0, 1],
          'run': [0, 1, 0],
          'tackle': [0, 0, 1]}
    """

    read_N = [32, 64, 96, 128]
    fft_n = [128]
    wind_f = ['hanning']

    def make(N, wf, rn):
        for xl in xls:
            invecs = _sample_xlsx(xl.path, sample_cnt, xl.sheets, xl.col, 2, rn, N, 0, True)
            #invecs /= np.max(invecs, axis=1)[:, np.newaxis]
            yield xl.label, fftn(invecs, N, wf=wf, fs=100, freq=True)

    plt.hold(True)
    for N in fft_n:
        for wf in wind_f:
            for rn in read_N:
                plt.figure()
                plt.grid()
                plt.xlabel('Frequency[Hz]')
                plt.ylabel('Power')
                for name, (fftdata, freq) in make(N, wf, rn):
                    plt.plot(freq, fftdata[0], color=fc[name], label=name)
                plt.legend()
                s = T('spectrumtest7/spectorum_{}p_{}-wf_{}-len.png'.format(N, wf, rn), mkdir=True)
                plt.savefig(s)
