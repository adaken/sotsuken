# coding: utf-8

from app.util import *
from app import R, T, L
from collections import namedtuple

if __name__ == '__main__':
    sample_cnt = 100

    Xl = namedtuple('Xl', 'path, sheets, col, label')
    xls = [Xl(R('data/raw/invectest/jump.xlsx'), ['Sheet'], 'A', 'jump'),
           Xl(R('data/raw/invectest/run.xlsx'), ['Sheet6', 'Sheet5', 'Sheet4'], 'F', 'run'),
           Xl(R('data/raw/invectest/walk.xlsx'), ['Sheet4'], 'F', 'walk')]
    for xl in xls:
        acc = ExcelWrapper(xl.path)[xl.sheets[0]].get_col(xl.col, (2, 129))
        print len(acc)
        m = fftn(acc, 128, savepath=L('action_fft_{}.png'.format(xl.label)), fs=100)
        print len(m)
        print m