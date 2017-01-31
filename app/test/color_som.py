# coding: utf-8

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from app.som.modsom import SOM
    from itertools import product, chain
    plt.hold(False)

    colors = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0],
                       [0, 1, 1], [1, 0, 0], [1, 0, 1],
                       [1, 1, 0], [1, 1, 1]])

    labels = ['black', 'blue', 'green',
              'aqua', 'red', 'purple',
              'yellow', 'white']

    font_colors = np.r_[colors[:-1], np.array([[0, 0, 0]])]

    reverse = lambda rgb: np.ones(3) if not np.any(rgb) else np.zeros(3) if np.all(rgb) else np.min(rgb) + np.max(rgb) - rgb

    invec = np.repeat(colors, 60, axis=0)
    np.random.shuffle(invec)

    plt.imshow(invec.reshape(20, 24, 3), interpolation='nearest')
    plt.axis('off')
    plt.savefig(r'D:\home\desk\colormap.png')

    som = SOM(invec, None, (20, 24), None)
    m  = som.train(1000)
    plt.imshow(m, interpolation='nearest')
    plt.axis('off')
    plt.savefig(r'D:\home\desk\colormap_som.png')

    g = SOM.to_umatrix(m)
    plt.imshow(g, interpolation='nearest', cmap='gray')
    plt.axis('off')
    plt.savefig(r'D:\home\desk\graymap_som.png')

    for v, l, fc in zip(colors, labels, font_colors):
        y, x = som._get_winner_node(v)
        plt.annotate(l, xy=(x, y), color=fc, fontsize=22, ha='center', va='center', fontweight='bold')
    plt.savefig(r'D:\home\desk\labeld_map.png')
