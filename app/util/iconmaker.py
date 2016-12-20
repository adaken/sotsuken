# coding: utf-8

from PIL import Image
from PIL import ImageDraw
import os
import random

def drow_circle(rgb, size, savepath):
    """
    円の画像(png)を作成

    Parameters
    ----------
    rgb : tuple or list
        円の色
    size : tuple or list
        円のサイズ
    savepath : str
        画像を保存するパス
    """

    assert isinstance(rgb, (tuple, list))
    assert isinstance(size, (tuple, list))
    assert isinstance(savepath, str)
    if os.path.exists(savepath): print "すでにファイルが存在するため上書きします: {}".format(savepath)
    if isinstance(rgb, list): rgb = tuple(rgb)
    if isinstance(size, list): size = tuple(size)
    im= Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse(((1, 1), size), outline=None, fill=rgb)
    del draw
    im.save(savepath)
    return savepath

def drow_random_color_circle(size, savepath):
    """
    ランダムな色の円(png)を作成

    Parameters
    ----------
    size : tuple or list
        円のサイズ
    savepath : str
        画像を保存するパス
    """
    rgb = tuple([random.randint(0, 255) for i in range(3)])
    return drow_circle(rgb, size, savepath)

if __name__ == '__main__':
    pass