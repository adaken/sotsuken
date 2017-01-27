# coding: utf-8

from app.util import timecounter, ExcelWrapper, fftn, drow_circle
from app.util import scale_zero_one, get_iter_len
from animationkml import ActionAnimationKml
from collections import namedtuple
from kmlwrapper import KmlWrapper
from datetime import timedelta
from sklearn.externals import joblib

def make_kml_with_acts(savename, anime_kml, kml_cnf, features, model,
                       act_icons=None, sample_n=128, subfeat=None):
    """アクションを使用したkmlを作成

    :param savename : str
    :param anime_kml : animationkml.AnimationKml
    :param kml_cnf : animationkml.KmlConfig
    :param features iterable of ndarray
    :param model : str
    :param act_icons : dict or None, default: None
    """

    acts = make_acts(features, model) # アクションのiterator
    len_, acts = get_iter_len(acts)
    acts = adjust_acts(acts, len_, sample_n) # 長さを調整

    # kml作成
    ak = ActionAnimationKml.from_anime_kml(anime_kml, acts=acts,
                                           act_icons=act_icons)
    ak.to_animatable(savename, kml_cnf)
    print "saved kml at {}".format(savename)

def make_acts(features, model, subfeat=None):
    """特徴ベクトルを訓練されたモデルで予測"""
    
    clf = joblib.load(model) # モデルをロード
    labels = clf.predict(features) # 多クラス分類器による識別
    return labels

def adjust_acts(acts, gps_len, sample_num, gps_hz=15, acc_hz=100):
    """アクションのリストの長さをGPSデータと調整してイテレート

    :param acts : iterable
    :param gps_len : int
    :return act_gen : generator
    """

    i, gps_hz, acc_hz = 0, float(gps_hz), float(acc_hz)
    gtsum = gps_T = 1 / gps_hz # GPSの周期
    atsum = act_T = 1 / acc_hz * sample_num # アクションの周期

    for act in acts:          # アクション1つにつき
        while gtsum <= atsum: # GPSの周期がアクションの周期内なら
            #print gtsum, atsum
            yield act         # 同じアクションを返す
            gtsum += gps_T    # GPSの時間を進める
            i += 1
        atsum += act_T        # 次のアクションの周期
    else: # 残りは最後のアクションで埋める
        #print "act_len:", i
        #print "gps_len:", gps_len
        sub = gps_len - i # GPSの長さとの差
        #print "sub:", sub
        for _ in xrange(sub): yield act

@timecounter
def make_kml_with_act():
    Xl = namedtuple('Xl', 'path, sheet, cols, begin')
    acc_xl = Xl(r'E:\work\data\acc_random_1206.xlsx', 'Sheet6', {'time':'A', 'acc':'F'},            2)
    gps_xl = Xl(r'E:\work\data\gps_random_1206.xlsx', 'Sheet3', {'time':'A', 'lat':'J', 'lon':'K'}, 9)
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
    sampling_step = 5

    def make_classed_acts(accs, act_names, N):
        """svmによって加速度リストからアクションリストを作成

        :param accs : 要素N個ごとに分割された加速度のリストのイテレータ
        :return acts : 長さlen(accs)のアクションのリスト
        """

        vecs = []
        ret = []
        for acc in accs:
            vec = scale_zero_one(fftn(acc, N))
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
    if len(classed_acts) < len(acc_times):
        acc_times.pop() # 余計な最後の要素を削除
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
                              act_icons=act_icons, sampling_step=sampling_step, icon_scale=0.4)

    print "kmlを作成しました: {}".format(save_path)

if __name__ == '__main__':
    from app.util import iter_acc_json, iter_gps_json
    from animationkml import KmlConfig
    from app.util import split_nlist
    from app import R, T, L
    from animationkml import AnimationKml

    def main():
        times, lats, lons = iter_gps_json(R('data/gps/gps_random_1206.json'))
        _, acc= iter_acc_json(R('data/acc/acc_random_1206.json'), prop=False)
        features = scale_zero_one(fftn(split_nlist(acc, 128), 128), axis=1)
        act_icons = {'run': T('run.png'),
                     'pk': T('pk.png'),
                     'tackle': T('tackle.png')}
        make_kml_with_acts(L('make_kml_test.kmz'),
                           AnimationKml(times, lats, lons),
                           kml_cnf=KmlConfig(0.5, 5, True),
                           features=features,
                           model=R('misc/model/clf2.pkl'),
                           act_icons=act_icons)

    def main2():
        pass
        #times, lats, lons =
    main()
