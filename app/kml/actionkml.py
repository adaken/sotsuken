# coding: utf-8

import simplekml as sk
import os
from itertools import islice
from app.util import drow_random_color_circle
from app import T

class AnimeKml(object):
    log = False

    def __init__(self, times, lons, lats, acts=None):
        """コンストラクタ

        :param times : iterable of daytimes or strs
            時間

        :param lons : iterable of floats
            'times'に対応する経度

        :param lats : iterable of floats
            'times'に対応する緯度

        :param acts : iterable of strs, default: None, optional
            'timesに対応するアクション名のリスト

        """

        self.times = self._format_times(times)
        self.lons = lons
        self.lats = lats
        self.acts = acts

    def to_animatable(self, savename, icon=None, iconscale=0.5, sampling_step=1,
                      kmz=False):
        """アニメーション可能なkmlを作成

        :param savename : str
            作成したファイルを保存するパス

        :param icon : str or None, default: None
            GoogleEarth上に表示される画像ファイルのパス、またはURL
            Noneの場合は'http://maps.google.com/mapfiles/kml/shapes/man.png'
            をダウンロード

        :param iconscale : int or float, default: 0.5
            アイコンのサイズ

        :param sampling_step : int, default: 1
            データに対するサンプリングの頻度

        :param kmz : bool
            True : kmzファイルを作成
            False: kmlファイルを作成
            kmzの場合は'icon'に指定したファイルがkmzに埋め込まれる
            kmlの場合は'icon'に指定したパスのみがkmlに埋め込まれる

        """

        if icon is None:
            icon = u'http://maps.google.com/mapfiles/kml/shapes/man.png'
        else:
            self._validate_res_available(icon)

        kml = sk.Kml()
        hotspot = sk.HotSpot(x=0, xunits=sk.Units.fraction,
                             y=0, yunits=sk.Units.fraction)

        # <IconStyle>
        iconstyle = sk.IconStyle(icon=sk.Icon(href=icon),
                                 scale=iconscale, hotspot=hotspot)
        sharedstyle = sk.Style(iconstyle=iconstyle) # 共通の<Style>

        iter_ = self.times, self.lons, self.lats
        range_ = 0, None, sampling_step
        for i, (t, lon, lat) in enumerate(islice(zip(*iter_), *range_)):

            # <Placemark> <Point> <coordinates>
            pnt = kml.newpoint(coords=[(lon, lat)])
            pnt.timestamp.when = t # <TimeStamp> <when>
            pnt.style = sharedstyle
        else:
            print "sum length of kml data:", i + 1

        if kmz:
            kml.savekmz(savename, True)
            print "saved as kmz"
        else:
            kml.save(savename, True)
            print "saved as kml"

    def to_act_animation(self, savename, act_icons=None, iconscale=0.5,
                         sampling_step=1, kmz=False):
        """動作ラベルを使ってアニメーション可能なkmlを作成

        このメソッドは呼び出す場合、コンストラクタに引数'acts'が渡されている必
        要がある

        :param savename : str
            作成したファイルを保存するパス

        :param act_icons : dict of str or None
            それぞれのアクションに対応するアイコンの辞書
            {'action': 'resorce of icon'}
            Noneの場合はランダムな画像が生成される

        :param iconscale : int or float, default: 0.5
            アイコンのサイズ

        :param sampling_step : int, default: 1
            データに対するサンプリングの頻度

        :param kmz : bool
            True : kmzファイルを作成
            False: kmlファイルを作成
            kmzの場合は'icon'に指定したファイルがkmzに埋め込まれる
            kmlの場合は'icon'に指定したパスのみがkmlに埋め込まれる

        """

        if self.acts is None:
            raise ValueError(u"this AnimeKml instance not given argument " \
                             u"'acts'")

        if act_icons is None:
            act_icons = {a: drow_random_color_circle(
                (16, 16), T('{}.png'.format(a))) for a in set(self.acts)}

        kml = sk.Kml()
        hotspot = sk.HotSpot(x=0, xunits=sk.Units.fraction,
                             y=0, yunits=sk.Units.fraction)

        # <IconStyle>
        sharedstyles = {act:sk.Style(sk.IconStyle(icon=sk.Icon(href=icon),
                                                  scale=iconscale,
                                                  hotspot=hotspot))
                        for act, icon in act_icons.items()}

        iter_ = self.times, self.lons, self.lats, self.acts
        range_ = 0, None, sampling_step
        for i, (t, lon, lat, act) in enumerate(islice(zip(*iter_)), *range_):

            # <Placemark> <Point> <coordinates>
            pnt = kml.newpoint(coords=[(lon, lat)])
            pnt.style = sharedstyles[act]
            pnt.timestamp.when = t # <TimeStamp> <when>
        else:
            print "sum length of kml data:", i + 1

        if kmz:
            kml.savekmz(savename, True)
            print "saved as kmz"
        else:
            kml.save(savename, True)
            print "saved as kml"

    @classmethod
    def _format_times(cls, times):
        """タイムスタンプの形式を整える

        yyyy/mm/hh:mm:ss.mmm -> yyyy-mm-dd"T"hh:mm:ss.mmm"Z"

        """

        format_new = "%Y-%m-%dT%H:%M:%S.%fZ" # 新しい形式
        return (t.strftime(format_new) for t in times)

    @classmethod
    def _validate_res_available(cls, *res):
        """リソースが利用可能かチェック"""

        def validate(res):
            if not isinstance(res, (str, unicode)):
                raise TypeError(u"invalid resource type: {}"
                                .format(res.__class__.name))

            # URLなら何もしない
            if len(res) >= 7:
                if res[:7] == 'http://' or res[:8] == 'https://':
                    return

            # URL以外ならチェック
            if not os.path.exists(res):
                raise ValueError(u"resource is not available: {}".format(res))
        map(validate, res)
