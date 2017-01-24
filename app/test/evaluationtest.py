# coding: utf-8

import os
from app import R, L, T
from app.util import *
from app.kml.kmlcreator import make_kml_with_acts
from app.kml.animationkml import AnimationKml, KmlConfig


def test():
    cnf = KmlConfig(iconscale=1, sampling_step=3, kmz=True)
    icon = lambda l, i: l + '_{}.png'.format(i)
    model = R('misc/model/Linear_A.pkl')
    m = os.path.splitext(os.path.basename(model))[0]
    readn = 128
    overlap = 0
    iconbase = R('img/icons/').p

    def accgps():
        gpsxls = R('data/gps/eval').ls(absp=True)[1]
        accxls = R('data/acc/eval').ls(absp=True)[1]

        def timecoord(gx, sheet, cols=('Time', 'Latitude', 'Longitude')):
            ws = ExcelWrapper(gx)[sheet]
            letters, rowidx = ws.find_letters_by_header(*cols)
            return ws.iter_cols(letters, (rowidx+1, None))

        def acc(ax, sheet):
            #ws = ExcelWrapper(ax)[sheet]
            #letter, rowidx = ws.find_letter_by_header('Magnitude Vector')
            a = make_input(xlsx=ax, sample_cnt=None, fft_N=128, read_N=readn,
                           sheetnames=[sheet],
                           normalizing='01', overlap=overlap)
            return a
            #return list(ws.iter_part_col(letter, 128, (rowidx+1, None)))

        for gx, ax in zip(gpsxls, accxls):
            print "read", gx, ax
            i = os.path.basename(os.path.splitext(gx)[0])
            times, lats, lons = timecoord(gx, 'Sheet1')
            acc_ = acc(ax, 'Sheet2')
            yield i, times, lats, lons, acc_

    def icons(i):
        return {'run':    iconbase + '\\' + icon('run', i),
                'pkick':  iconbase + '\\' + icon('pkick', i),
                'tackle': iconbase + '\\' + icon('tackle', i),
                'pass':   iconbase + '\\' + icon('pass', i),
                'walk':   iconbase + '\\' + icon('walk', i)}

    for i, times, lats, lons, acc in accgps():
        #feats = fftn(acc, fft_N=128, wf='hanning', fs=100)
        feats = acc
        print len(times), len(lats), len(lons), len(acc)
        anime_kml = AnimationKml(times, lats, lons)
        make_kml_with_acts(L('evaltest/test_new_{}_{}len_{}_{}overlap.kmz'.format(m, readn, i, overlap), mkdir=True),
                           anime_kml, kml_cnf=cnf, features=feats,
                           model=model, act_icons=icons(i), sample_n=readn - overlap)

if __name__ == '__main__':
    test()