# coding: utf-8

from app.util import fftn
from app.util import scale_zero_one, get_iter_len
from animationkml import ActionAnimationKml
from sklearn.externals import joblib
from app import R, T, L
import numpy as np

def make_kml_with_acts(savename, anime_kml, kml_cnf, features, model,
                       act_icons=None, sample_n=128):
    """アクションを使用したkmlを作成

    :param savename : str
    :param anime_kml : animationkml.AnimationKml
    :param kml_cnf : animationkml.KmlConfig
    :param features iterable of ndarray
    :param model : str
    :param act_icons : dict or None, default: None
    """

    acts = make_acts(features, model) # アクションのiterator
    #acts = make_acts2(features, vs='VS', p=32) # アクションのiterator
    len_, acts = get_iter_len(acts)
    acts = adjust_acts(acts, len_, sample_n) # 長さを調整

    # kml作成
    ak = ActionAnimationKml.from_anime_kml(anime_kml, acts=acts,
                                           act_icons=act_icons)
    ak.to_animatable(savename, kml_cnf)
    print "saved kml at {}".format(savename)

def make_acts(features, model):
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
    #main()
