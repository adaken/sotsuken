# coding: utf-8
from collections import namedtuple
from app.util.inputmaker import make_input
from modsom import SOM
from app import R, T, L
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.cm as cm

def som_json(jsons, labels, label_colors=None, train_cnt=50, mapsize=None):
    """jsonからsom

    :param jsons : list of str
        特徴ベクトルのみのjsonのパスのリスト

    :param labels : list
        それぞれのjsonに対応するラベル

    :param label_colors : dict, None
        ラベルに対応するマップに表示する際の文字色
    """

    if label_colors is None:
        len_ = len(labels)
        label_colors = {str(l): cm.autumn(float(i)/len_)
                        for i, l in enumerate(labels)}

    input_ = []
    labels_ = []

    for json_, label in zip(jsons, labels):
        with open(json_) as fp:
            features = json.load(fp)
            input_ += features
            labels_ += [label] * len(features)

    if mapsize is None:
        som = SOM(input_, labels_, display='um')
    else:
        som = SOM(input_, labels_, shape=mapsize, display='um')

    map_, labels, coords = som.train(train_cnt)
    plt.imshow(map_, interpolation='nearest', cmap='gray')
    for label, coord in zip(labels, coords):
        x, y = coord
        plt.text(x, y, label, color=label_colors[label])
    plt.show()

if __name__ == '__main__':
    from app import R
    som_json([R('data/acc/fft/placekick_acc_128p_52data.json'),
              R('data/acc/fft/run_acc_128p_132data.json'),
              R('data/acc/fft/tackle_acc_128p_92data.json'),
              #R('data/acc/fft/')
              ],
             ['P', 'R', 'T'], train_cnt=1000)
