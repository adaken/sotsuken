# coding: utf-8

from app.util import *
from app.som.modsom import SOM
from collections import namedtuple
from app import R, T, L
import matplotlib.pyplot as plt
plt.hold(False)

if __name__ == '__main__':
    train_cnt = 300
    map_size = (40, 50)
    sample_cnt = 100

    Xl = namedtuple('Xl', 'path, sheets, col, label')
    xls = [Xl(R('data/raw/invectest/jump.xlsx'), ['Sheet'], 'A', 'jump'),
           Xl(R('data/raw/invectest/run.xlsx'), ['Sheet6', 'Sheet5', 'Sheet4'], 'F', 'run'),
           Xl(R('data/raw/invectest/walk.xlsx'), ['Sheet4', 'Sheet1'], 'F', 'walk')]

    xls = [Xl(R('data/acc/pass_acc_128p_131data.xlsx'), ['Sheet1'], 'A', 'pass'),
           Xl(R('data/acc/placekick_acc_128p_101data.xlsx'), ['Sheet1'], 'A', 'pkick'),
           Xl(R('data/acc/run_acc_128p_132data.xlsx'), ['Sheet1'], 'A', 'run'),
           Xl(R('data/acc/tackle_acc_128p_111data.xlsx'), ['Sheet1'], 'A', 'tackle'),
           Xl(R('data/raw/invectest/walk.xlsx'), ['Sheet4', 'Sheet1'], 'F', 'walk')]

    fc = {'jump': [0, 1, 0],
          'run': [1, 0, 0],
          'walk': [0, 0, 1]}

    fc = {'pass': [1, 0, 1],
          'pkick': [0, 0, 1],
          'run': [0, 1, 0],
          'tackle': [1, 0, 0],
          'walk': [0, 1, 1]}

    #read_N = [32, 64, 96, 128]
    read_N = [96, 128]
    fft_n = [128]
    wind_f = ['hanning']

    def make(N, wf, rn):
        invec, labels = [], []
        for xl in xls:
            i, l = make_input(xl.path, sample_cnt, xl.sheets, col=xl.col,
                              read_N=rn, fft_N=N, wf=wf, label=xl.label, normalizing='01')
            invec += i.tolist()
            labels += l
        return invec, labels

    for i in xrange(2):
        for N in fft_n:
            for wf in wind_f:
                for rn in read_N:
                    invec, labels = make(N, wf, rn)
                    som = SOM(invec, labels, map_size)
                    map_, labels_, coords = som.train(train_cnt)
                    plt.imshow(map_, cmap='gray', interpolation='nearest')
                    for l, (c1, c2) in zip(labels_, coords):
                        s = T('invectest6_act_divide_on_max/{}_test_{}p_{}_{}veclen.png'.format(i, N, wf, rn), mkdir=True)
                        plt.text(c1, c2, l, color=fc[l], va='center', ha='center')
                    plt.axis('off')
                    plt.savefig(s)