# coding: utf-8

from kml.kmlwrapper import KmlWrapper
from util.excelwrapper import ExcelWrapper
from util.util import drow_circle
from collections import namedtuple

if __name__ == '__main__':

    Xl = namedtuple('Xl', 'path, id')
    xls = [Xl(r'E:\work\players\player_1_coord.xlsx', '1'),
           Xl(r'E:\work\players\player_2_coord.xlsx', '2'),
           Xl(r'E:\work\players\player_3_coord.xlsx', '3'),
           Xl(r'E:\work\players\player_4_coord.xlsx', '4'),
           Xl(r'E:\work\players\player_5_coord.xlsx', '5')]

    icon_colors = {'1': [255, 0, 0],
                   '2': [204, 86, 22],
                   '3': [245, 216, 0],
                   '4': [0, 255, 0],
                   '5': [0, 0, 255]}

    for i, xl in enumerate(xls):
        save = r'E:\work\players\kml\players{}.kml'.format(i+1)
        icon_res = drow_circle(icon_colors[xl.id], (8, 8), r'E:\tmp\players{}.png'.format(i+1))
        ws = ExcelWrapper(xl.path).get_sheet('Sheet1')
        times, lons, lats = ws.iter_cols(('A', 'K', 'J'), (9, None), log=True)
        KmlWrapper().createAnimeKml(save, times, lons, lats,
                                    icon_res=icon_res, icon_scale=0.5, sampling_step=5)