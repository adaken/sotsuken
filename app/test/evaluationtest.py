# coding: utf-8

import os
from app import R, L, T
from app.util import *
from app.kml.kmlcreator import make_kml_with_acts
from app.kml.animationkml import AnimationKml, KmlConfig


def test():
    cnf = KmlConfig(iconscale=1, sampling_step=3, kmz=True)
    icon = lambda l, i: l + '_{}.png'.format(i)
    model = R('misc/model/rbf_1k_1k-_VS_64p.pkl')
    #model = 'mod_Against_16_p.pkl'
    modelname = os.path.splitext(os.path.basename(model))[0]
    exfs = 64
    read_N = 64
    fft_N = 64
    overlap = 32
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
            print "xl:", ax
            print "sh:", sheet
            a = make_input(xlsx=ax, sample_cnt=None, exfs=exfs, fft_N=fft_N,
                           read_N=read_N, sheetnames=[sheet],
                           normalizing=None, overlap=overlap)
            return scale_zero_one(a)
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
        make_kml_with_acts(T('evaltest3/{}_{}-model_{}-vec_{}p-fft_{}-ovlap.kmz'
                             .format(i, modelname, read_N, fft_N, overlap), mkdir=True),
                           anime_kml, kml_cnf=cnf, features=feats,
                           model=model, act_icons=icons(i), sample_n=read_N - overlap)

def test2():
    cnf = KmlConfig(iconscale=1, sampling_step=3, kmz=True)
    icon = lambda l, i: l + '_{}.png'.format(i)
    #model = R('misc/model/Rbf_5class_VS_32p.pkl')
    model = 'mod3264p.pkl'
    modelname = os.path.splitext(os.path.basename(model))[0]
    exfs = (32, 64)
    read_N = (32, 64)
    fft_N = (32, 64)
    overlap = (0, 32)
    iconbase = R('img/icons/').p

    def accgps():
        gpsxls = R('data/gps/eval').ls(absp=True)[1]
        accxls = R('data/acc/eval').ls(absp=True)[1]

        def timecoord(gx, sheet, cols=('Time', 'Latitude', 'Longitude')):
            ws = ExcelWrapper(gx)[sheet]
            letters, rowidx = ws.find_letters_by_header(*cols)
            return ws.iter_cols(letters, (rowidx+1, None))

        def acc(ax, sheet):
            print "xl:", ax
            print "sh:", sheet
            ret = []
            for ex, rn, fn, ol in zip(exfs, read_N, fft_N, overlap):
                a = make_input(xlsx=ax, sample_cnt=None, exfs=ex, fft_N=fn,
                               read_N=rn, sheetnames=[sheet],
                               normalizing=None, overlap=ol)
                ret.append(scale_zero_one(a))
            return ret

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
        X, subX = acc
        print len(times), len(lats), len(lons), len(X), len(subX)
        anime_kml = AnimationKml(times, lats, lons)
        make_kml_with_acts(T('evaltest2/{}_{}-model_{}-vec_{}p-fft_{}-ovlap.kmz'
                             .format(i, modelname, read_N, fft_N, overlap),
                             mkdir=True),
                            anime_kml, kml_cnf=cnf, X=X,subX=subX,
                            act_icons=icons(i), sample_n=32)


if __name__ == '__main__':
    #test()
    test2()
