# coding: utf-8

from app.util import *
from app.som.modsom import SOM
from collections import namedtuple
from app import R, T, L
import matplotlib.pyplot as plt
import numpy as np
plt.hold(False)

if __name__ == '__main__':
    train_cnt = 500
    map_size = (40, 50)
    sample_cnt = 100

    Xl = namedtuple('Xl', 'path, sheets, col, label')

    """
    xls = [Xl(R('data/raw/invectest/jump.xlsx'), ['Sheet'], 'A', 'jump'),
           Xl(R('data/raw/invectest/run.xlsx'), ['Sheet6', 'Sheet5', 'Sheet4'], 'F', 'run'),
           Xl(R('data/raw/invectest/walk.xlsx'), ['Sheet4', 'Sheet1'], 'F', 'walk')]

    fc = {'jump': [1, 0, 0],
          'run': [0, 1, 0],
          'walk': [0, 0, 1]}

    """
    xls = [
        Xl(R('data/acc/pass_acc_128p_131data.xlsx'), ['Sheet1'], 'A', 'pass'),
        Xl(R('data/acc/placekick_acc_128p_101data.xlsx'), ['Sheet1'], 'A', 'pkick'),
        Xl(R('data/acc/run_acc_128p_132data.xlsx'), ['Sheet1'], 'A', 'run'),
        Xl(R('data/acc/tackle_acc_128p_111data.xlsx'), ['Sheet1'], 'A', 'tackle'),
        Xl(R('data/raw/invectest/walk.xlsx'), ['Sheet4', 'Sheet1'], 'F', 'walk')
        ]

    fc = {'pass': [1, 0, 1],
          'pkick': [0, 0, 1],
          'run': [0, 1, 0],
          'tackle': [1, 0, 0],
          'walk': [0, 1, 1]}

    #read_N = [32, 64, 96, 128]
    read_N = [96]
    #fft_N =  [32, 64, 96, 128]
    fft_N =  [128]
    wind_f = ['hanning']

    def make(N, wf, rn):
        invec, labels = [], []
        fig, ax = plt.subplots(nrows=2, ncols=3)
        ax = ax.ravel()
        for x, xl in enumerate(xls):
            i, l = make_input(xl.path, sample_cnt, xl.sheets, col=xl.col,
                              read_N=rn, fft_N=N, wf=wf, label=xl.label,
                              exfs=128, normalizing=None)
            invec += i.tolist()
            labels += l
            ax[x].set_title(xl.label)
            ax[x].hold(True)
            for j in i:
                ax[x].plot(j)
        invec = scale_zero_one(np.array(invec), None)
        fig.savefig(r'E:\test{}-rn.png'.format(rn))
        return invec, labels

    def do(i):
        for wf in wind_f:
            for N, rn in zip(fft_N, read_N):
                invec, labels = make(N, wf, rn)
                som = SOM(invec, labels, map_size)
                map_, labels_, coords = som.train(train_cnt)
                fig, ax = plt.subplots(1, 1)
                ax.imshow(map_, cmap='gray', interpolation='nearest')
                for l, (c1, c2) in zip(labels_, coords):
                    s = T('invectest/nmfix_{}_{}pfft_{}-w_{}-len_{}-loop.png'
                          .format(i, N, wf, rn, train_cnt), mkdir=True)
                    ax.text(c1, c2, l, color=fc[l], va='center', ha='center')
                ax.axis('off')
                fig.savefig(s)

    def func1():
        for i in xrange(1):
            do(i)

    def func2():
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from app.kml.kmlcreator import make_acts2
        n = 32
        X, L = make(n, 'hanning', n)
        P = make_acts2(np.array(X), vs='VS', p=n)
        print P
        print accuracy_score(L, P)
        print classification_report(L, P)
        print confusion_matrix(L, P)

    func1()
