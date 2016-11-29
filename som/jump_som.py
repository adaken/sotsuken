# coding: utf-8
from collections import namedtuple

if __name__ == '__main__':
    Xl = namedtuple('Xl', 'filename, sheets, letter, color, sampling, overlap')
    xls =  {
        'R':Xl(r'E:\work\data\new_run.xlsx', ('Sheet4', 'Sheet5', 'Sheet6'), 'F', 'red', 'rand', 0),
        'Jg':Xl(r'E:\work\data\jog_jump.xlsx', ('Sheet4', 'Sheet5', 'Sheet6'), 'F', 'green', 'std', 64),
        'W':Xl(r'E:\work\data\walk_jump.xlsx', ('Sheet4', 'Sheet5', 'Sheet6'), 'F', 'blue', 'std', 64),
        'Jp':Xl(r'E:\work\data\jump_128p_84data_fixed.xlsx', ('Sheet',), 'A', 'deeppink', 'std', 0),
        #'S':Xl(r'E:\work\data\skip.xlsx', ('Sheet4',), 'F', 'purple', 'rand', 0)
        }
    from util.util import make_input_from_xlsx
    import numpy as np
    input_data = []
    for label, xl in xls.items():
        r = np.random.randint(len(xl.sheets))
        r = 0
        print "Sheetname:", xl.sheets[r]
        vecs_with_label = make_input_from_xlsx(filename=xl.filename, sheetname=xl.sheets[r],
                                               col=xl.letter, read_range=(2, None), overlap=xl.overlap,
                                               sampling=xl.sampling, sample_cnt=84, fft_N=128,
                                               normalizing='01', label=label, log=False)
        input_data += vecs_with_label
    from modsom import SOM
    som = SOM(shape=(40, 50), input_data=input_data, display='gray_scale')
    map_, label_with_coords = som.train(200)
    import matplotlib.pyplot as plt
    plt.hold(False)
    plt.imshow(map_, interpolation='nearest')
    for label, coord in label_with_coords:
        x, y = coord
        plt.text(x, y, label, color=xls[label].color)
    import random
    rs = 'abcdefghijklmnopqrstuvwxyz0123456789'
    plt.savefig(r'E:\work\fig\run-jog-walk-jump_som\fig_200roop_std_{}.png'
                .format("".join(random.choice(rs) for i in range(8))))
