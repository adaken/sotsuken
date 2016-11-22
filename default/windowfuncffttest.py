# coding: utf-8
from collections import namedtuple

if __name__ == '__main__':
    from util.util import make_input_from_xlsx
    from som.modsom import SOM
    import matplotlib.pyplot as plt
    plt.hold(False)
    Xl = namedtuple('Xl', 'path, sheet, letter, sampling, color')
    wfs = ['hunning', 'humming', 'blackman']
    xls = {'R' :Xl(r'E:\work\data\new_run.xlsx',                'Sheet4', 'F', 'rand', 'red'),
           'W':Xl(r'E:\work\data\walk.xlsx',                   'Sheet4', 'F', 'std',  'blue'),
           'J':Xl(r'E:\work\data\jump_128p_84data_fixed.xlsx', 'Sheet',  'A', 'std',  'green')}

    def make_input(wf):
        in_vec = []
        for label, xl in xls.items():
            v = make_input_from_xlsx(filename=xl.path, sheetname=xl.sheet, col=xl.letter,
                                     read_range=(2, None), sampling=xl.sampling, sample_cnt=84,
                                     overlap=0, fft_N=128, fft_wf=wf,
                                     normalizing='01', label=label, log=False)
            in_vec += v
        return in_vec

    def make_fig(i):
        for wf in wfs:
            print "wf:", wf
            in_vec = make_input(wf)
            som = SOM(shape=(40, 50), input_data=in_vec, display='gray_scale')
            map_, lc = som.train(200)
            plt.imshow(map_, interpolation='nearest')
            for l, c in lc:
                x, y = c
                plt.text(x, y, l, color=xls[l].color)
            plt.savefig(r'E:\work\fig\som_window_func_test\run_walk_jump_{}_window{}'.format(wf, i))

    for i in range(5):
        make_fig(i)

