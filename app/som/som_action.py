# coding: utf-8
from collections import namedtuple
from app.util.inputmaker import make_input
from modsom import SOM
from app import R, T, L
import matplotlib.pyplot as plt

def som_action():
    sample_cnt = 90
    min_row = 2
    N = 128
    Xl = namedtuple('Xl', 'path, sheet, col')
    xls = {'S':Xl(R(r'data\raw\acc_stop_1206.xlsx'), 'Sheet4', 'F'),
           'B':Xl(R(r'data\raw\walk.xlsx'), 'Sheet4', 'F'),
           'R':Xl(R(r'data\raw\run_1122.xlsx'), 'Sheet4', 'F'),
           'J':Xl(R(r'data\raw\jump_128p_174data_fixed.xlsx'), 'Sheet', 'A')}

    colors = {'S':'blue',
              'B':'green',
              'R':'red',
              'J':'deeppink'}

    in_vecs = []
    for key, xl in xls.items():
        vecs, labels = make_input(xlsx=xl.path, sheetnames=[xl.sheet], col=xl.col,
                                  min_row=min_row, fft_N=N, sample_cnt=sample_cnt,
                                  label=key, wf='hanning', normalizing='std', sampling='std',
                                  overlap=0, log=True)
        vecs = [[key, vec] for vec in vecs]
        in_vecs += vecs

    som = SOM(shape=(40, 60), input_data=in_vecs, display='gray_scale')
    map_, labels = som.train(200)
    plt.imshow(map_, interpolation='nearest')
    for label, coord in labels:
        x, y = coord
        plt.text(x, y, label, color=colors[label])
    plt.show()

if __name__ == '__main__':
    som_action()
