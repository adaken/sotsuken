# coding: utf-8
from collections import namedtuple
from app import R, T, L
from app.util.inputmaker import make_input
from app.util.inputmaker import random_input_iter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from app.som.modsom import SOM
    Xl = namedtuple('Xl', 'path, sheet, col, label, color')
    xls = [Xl(R('data/acc/tackle_acc_128p_62data.xlsx'), 'Sheet', 'A', 'T','yellow'),
           Xl(R('data/acc/run_acc_128p_81data.xlsx'), 'Sheet2', 'F', 'R', 'red'),
           Xl(R('data/acc/place.kick_128p_22data.xlsx'), 'Sheet', 'A', 'P', 'blue'),
           Xl(R('data/acc/dropkick_acc_128p_16data.xlsx'), 'Sheet', 'A', 'D', 'green')]

    colors = {xl.label : xl.color for xl in xls}

    inputvec = []
    inputlabel = []
    for xl in xls:
        v, l = make_input(xl.path, [xl.sheet], xl.col, min_row=2, fft_N=128, sample_cnt=16,
                       label=xl.label, normalizing='std', log=True)
        for i in v: inputvec.append(i)
        inputlabel += l
    inputdata = [(l, v) for v, l in random_input_iter(inputvec, inputlabel)]

    som = SOM((60, 40), inputdata, display='gray_scale')
    map_, label_coords = som.train(1000)

    plt.imshow(map_)
    for l, c in label_coords:
        x, y = c
        plt.text(x, y, l, color=colors[l])
    plt.show()