# coding: utf-8
from collections import namedtuple
from app import R, T, L

if __name__ == '__main__':
    from app.som.modsom import SOM
    Xl = namedtuple('Xl', 'path, sheet')
    xls = [Xl(R())]
    som = SOM((60, 40), input_data, display='gray_scale')