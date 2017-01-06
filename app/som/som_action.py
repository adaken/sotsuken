# coding: utf-8
from collections import namedtuple
from app.util.inputmaker import make_input
from modsom import SOM
from app import R, T, L
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.cm as cm
from itertools import chain, product, izip
from app.util.jsonio import iter_inputs_json

def som_json(jsons, labels, label_colors=None, train_cnt=50, mapsize=None):
    """jsonからsom

    :param jsons : list of str
        特徴ベクトルのみのjsonのパスのリスト

    :param labels : list
        マップに表示するそれぞれのjsonに対応するラベル

    :param label_colors : dict, None
        ラベルに対応するマップに表示する際の文字色
    """

    inputs = []
    for j, label in zip(jsons, labels):
        _, f_iter  = iter_inputs_json(j, True)
        inputs = chain(inputs, product(f_iter, [label]))

    if label_colors is None:
        len_ = len(labels)
        label_colors = {str(l): cm.autumn(float(i)/len_)
                        for i, l in enumerate(labels)}

    features, labels = zip(*inputs)
    features = np.array(features)
    labels = list(labels)

    if mapsize is None:
        som = SOM(features, labels, display='um')
    else:
        som = SOM(features, labels, shape=mapsize, display='um')

    map_, labels, coords = som.train(train_cnt)
    plt.imshow(map_, interpolation='nearest', cmap='gray')
    for label, coord in zip(labels, coords):
        x, y = coord
        plt.text(x, y, label, color=label_colors[label])
    plt.show()

if __name__ == '__main__':
    from app import R
    som_json([
        R('data/fft/placekick_acc_128p_52data.json'),
        R('data/fft/run_acc_128p_132data.json'),
        R('data/fft/tackle_acc_128p_92data.json'),
        R('data/fft/pass_acc_128p_31data.json')
        ],
        ['PKick', 'Run', 'Tackle', 'Pass'], train_cnt=400)
