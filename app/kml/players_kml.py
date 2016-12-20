# coding: utf-8

from app.kml.kmlwrapper import KmlWrapper
from app.util.excelwrapper import ExcelWrapper
from app.util.iconmaker import drow_circle
from collections import namedtuple
from app import R, T

if __name__ == '__main__':

    Xl = namedtuple('Xl', 'path, id')
    xls = [Xl(R(r'data\gps\players\coord1.xlsx'), '1'),
           Xl(R(r'data\gps\players\coord2.xlsx'), '2'),
           Xl(R(r'data\gps\players\coord3.xlsx'), '3'),
           Xl(R(r'data\gps\players\coord4.xlsx'), '4'),
           Xl(R(r'data\gps\players\coord5.xlsx'), '5')]

    icon_colors = {'1': [255, 0, 0],
                   '2': [241, 147, 0],
                   '3': [245, 216, 0],
                   '4': [0, 255, 0],
                   '5': [0, 0, 255]}

    for i, xl in enumerate(xls):
        save = R(r'data\kml\players\{}.kml').format(i+1)
        icon_res = drow_circle(icon_colors[xl.id], (8, 8), T(r'players{}.png').format(i+1))
>>>>>>> refs/heads/test
        ws = ExcelWrapper(xl.path).get_sheet('Sheet1')
        times, lons, lats = ws.iter_cols(('A', 'K', 'J'), (9, None), log=True)
        KmlWrapper().createAnimeKml(save, times, lons, lats,
                                    icon_res=icon_res, icon_scale=0.3, sampling_step=10)
