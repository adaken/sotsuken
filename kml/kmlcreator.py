# coding: utf-8

from util.util import timecounter
from util.excelwrapper import ExcelWrapper
from util.fft import fft
from collections import namedtuple
from kmlwrapper import KmlWrapper
from util.util import drow_circle
from datetime import timedelta
from util.util import normalize_scale
import random

@timecounter
def make_kml_with_act():
    Xl = namedtuple('Xl', 'path, sheet, cols, begin')
    acc_xl = Xl(r'E:\work\data\acc_random_1206.xlsx', 'Sheet4', {'time':'A', 'acc':'F'},            2)
    gps_xl = Xl(r'E:\work\data\gps_random_1206.xlsx', 'Sheet1', {'time':'A', 'lat':'J', 'lon':'K'}, 9)
    N = 128

    save_path = r'E:\kml_act_test.kml'
    icon_size = (16, 16)
    act_icons = {'run' :drow_circle(rgb=(255, 0, 0), size=icon_size, savepath=r'E:\tmp\run.png'),
                 'jump':drow_circle(rgb=(0, 255, 0), size=icon_size, savepath=r'E:\tmp\jump.png'),
                 'walk':drow_circle(rgb=(0, 0, 255), size=icon_size, savepath=r'E:\tmp\walk.png')}
    act_names = act_icons.keys()

    gps_ws = ExcelWrapper(gps_xl.path).get_sheet(gps_xl.sheet)
    times = gps_ws.get_col(gps_xl.cols['time'], (gps_xl.begin, None))
    print >> open(r'E:\log_gps_times.txt', 'w'), times
    lats  = gps_ws.get_col(gps_xl.cols['lat'], (gps_xl.begin, None))
    lons  = gps_ws.get_col(gps_xl.cols['lon'], (gps_xl.begin, None))
    sampling_step = 1

    def make_classed_acts(accs, act_names, N):
        """svmによって加速度リストからアクションリストを作成

        :param accs : 要素N個ごとに分割された加速度のリストのイテレータ
        :return acts : 長さlen(accs)のアクションのリスト
        """

        vecs = []
        ret = []
        for acc in accs:
            vec = normalize_scale(fft(acc, N))
            vecs.append(vec)

        from sklearn.externals import joblib
        clf = joblib.load('E:\clf.pkl')
        pred = clf.predict(vecs)  #他クラス分類器による識別
        #ret.append(i for i in list(pred))
        map(ret.append, pred)

        print pred

        print len(times),len(ret)
        diff= int(len(times)/15/1.28) - len(ret)
        print diff
        if (diff != 0):
            #ret.append(ret[-1]*diff)# リストの長さを調整
            for i in xrange(diff): ret.append(ret[-1])

        """
        # テスト用コード
        ret = []
        ret = [act_names[random.randint(0, len(act_names) - 1)] for i in xrange(len(accs))]
        """
        return ret

    acc_ws = ExcelWrapper(acc_xl.path).get_sheet(acc_xl.sheet)
    classed_acts = make_classed_acts(list(acc_ws.iter_part_col(acc_xl.cols['acc'], N, (acc_xl.begin, None))), act_names, N)
    print >> open(r'E:\log_classed_acts.txt', 'w'), classed_acts
    acc_times = acc_ws.get_col(acc_xl.cols['time'], (acc_xl.begin, None), iter_cell=False, log=True)[::N]
    print >> open(r'E:\log_acc_times.txt', 'w'), acc_times
    #acc_times.pop() # 余計な最後の要素を削除
    assert len(classed_acts) == len(acc_times), "act: {}, times: {}".format(len(classed_acts), len(acc_times))

    @timecounter
    def make_acts(gps_times, acts, acc_times):
        """アクションのリストを作成

        時刻を比較し、差が最も少ない時刻のアクションをリスト化

        """

        assert len(acts) == len(acc_times)
        logf_path = r'E:\log_make-acts().txt'
        with open(logf_path, 'w'): pass
        logf = open(logf_path, 'a')
        ret = []
        max_i = len(acts)
        print >> logf, "max_i        :", max_i
        print >> logf, "gps_times_len:", len(gps_times), "\n"
        for gt in gps_times:
            i = 0
            p_delta = timedelta.max
            p_act = None
            while i < max_i:
                c_delta = gt - acc_times[i]
                c_delta = abs(c_delta)
                print >> logf, "acc_times_index:", i
                print >> logf, "previous_delta :", p_delta
                print >> logf, "current_delta  :", c_delta
                if (c_delta < p_delta) and (i != max_i - 1):
                    print >> logf, "最小誤差更新\n"
                    p_delta = c_delta
                    p_act = acts[i]
                    i += 1
                else:
                    print >> logf, "更新なし"
                    print >> logf, "add_action:", p_act, '\n'
                    ret.append(p_act)
                    break

        #ret.append(ret[-1]) # リストの長さを調整
        print >> open(r'E:\log_acts.txt', 'w'), ret
        return ret

    def make_acts2(gps_times, acts):
        ret = []
        #map(ret.append, acts[int(gps_times/15/1.28)])
        
        for i in xrange(len(gps_times) - 1):
            idx = int(i/15/1.28)
            ret.append(acts[idx])
        
        return ret
        
    #acts  = make_acts(times, classed_acts, acc_times)
    acts = make_acts2(times, classed_acts)

    # kml生成
    KmlWrapper().createAnimeKml(save_path, times, lons, lats, acts=acts,
                              act_icons=act_icons, sampling_step=sampling_step, icon_scale=0.5)

    print "kmlを作成しました: {}".format(save_path)

if __name__ == '__main__':

    make_kml_with_act()

    def main():

        # リソース
        xlsx_path = r"E:\work\bicycle_gps_hirano.xlsx"
        kml_path = r"E:\work\out_bicycle.kml"
        icon_res = r"E:\work\circle_blue.png"

        # Excelを読み込む
        from util.excelwrapper import ExcelWrapper
        ws = ExcelWrapper(filename=xlsx_path, sheetname='Sheet1')

        # 読み込みを開始する行
        begin_row = 9

        # 列のリストを取得する関数
        getcol = lambda l : ws.select_column(col_letter=l, begin_row=begin_row)

        # kmlに書き出す
        from kml.kmlwrapper import KmlWrapper
        KmlWrapper().createAnimeKml(save_path=kml_path, times=getcol('A'), longitudes=getcol('K'),
                                  latitudes=getcol('J'), format_time=True, sampling_interval=15,
                                  icon_res=icon_res, icon_scale=0.6)
        print "completed!"
