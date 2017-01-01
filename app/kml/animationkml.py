# coding: utf-8

import simplekml as sk
import os
from itertools import islice
from app.util import drow_random_color_circle
from app import T

class KmlConfig(object):
    """Kml作成の際のオプションを定義"""

    def __init__(self, iconscale=0.5, sampling_step=1, kmz=False,
                 hotspot=None):
        """Constructor

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

        self.iconscale = iconscale
        self.sampling_step = sampling_step
        self.kmz = kmz

        if hotspot is None:
            self.hotspot = sk.HotSpot(x=0, xunits=sk.Units.fraction,
                                      y=0, yunits=sk.Units.fraction)
        else:
            self.hotspot = hotspot

class AnimationKml(object):
    """抽象クラス"""

    def __init__(self, times, lats, lons):
        """Constructor

        :param times : iterable of daytimes or strs
            時間

        :param lats : iterable of floats
            'times'に対応する緯度

        :param lons : iterable of floats
            'times'に対応する経度
        """

        self.times = times
        self.lats = lats
        self.lons = lons

    def __call__(self, savename, kml_cnf):
        """Abstract method

        to_animatableへのアクセサ
        """

        self.to_animatable(savename, kml_cnf)

    def to_animatable(self, savename, kml_cnf=None):
        """Abstract method

        :param savename : str
            保存するパス

        :param kml_cnf : KmlConfig, default: None
            オプション
        """
        pass

    @classmethod
    def from_anime_kml(cls, anime_kml, **kwargs):
        """AnimationKmlのインスタンスからインスタンスを取得"""

        ak = anime_kml
        return cls(ak.times, ak.lats, ak.lons, **kwargs)

    @classmethod
    def _format_times(cls, times):
        """タイムスタンプの形式を整える

        yyyy/mm/hh:mm:ss.mmm -> yyyy-mm-dd"T"hh:mm:ss.mmm"Z"
        """

        format_new = "%Y-%m-%dT%H:%M:%S.%fZ" # 新しい形式
        print "formatting times..."
        return (t.strftime(format_new) for t in times)

    @classmethod
    def _validate_res_available(cls, *res):
        """リソースが利用可能かチェック"""

        def validate(res):
            if not isinstance(res, (str, unicode)):
                raise TypeError(u"invalid resource type: {}"
                                .format(res.__class__.name))

            if len(res) >= 7: # URLなら何もしない
                if res[:7] == 'http://' or res[:8] == 'https://':
                    return

            if not os.path.exists(res): # URL以外ならチェック
                raise ValueError(u"resource is not available: {}".format(res))
        map(validate, res)

    @classmethod
    def _parse_conf(self, kml_cnf):
        if kml_cnf is None:
            return KmlConfig()
        return kml_cnf

    @classmethod
    def _save(self, kml, savename, kmz):
        if kmz:
            kml.savekmz(savename, True)
            print "saved as kmz"
        else:
            kml.save(savename, True)
            print "saved as kml"

class SimpleAnimationKml(AnimationKml):
    """アニメーション可能なKmlを作成するクラス"""

    def __init__(self, times, lats, lons, icon=None):
        """Constructor

        :param icon : str or None, default: None
            GoogleEarth上に表示される画像ファイルのパス、またはURL
            Noneの場合は'http://maps.google.com/mapfiles/kml/shapes/man.png'
            をダウンロード

        See also AnimationKml.__init__()
        """
        super(SimpleAnimationKml, self).__init__(times, lats, lons)
        if icon is None:
            icon = u'http://maps.google.com/mapfiles/kml/shapes/man.png'
        self._validate_res_available(icon)
        self.icon = icon

    def to_animatable(self, savename, kml_cnf):
        """Override

        See AnimationKml.to_animatable()
        """

        cnf = self._parse_conf(kml_cnf)
        kml = sk.Kml()

        # <IconStyle>
        iconstyle = sk.IconStyle(icon=sk.Icon(href=self.icon),
                                 scale=cnf.iconscale, hotspot=cnf.hotspot)
        sharedstyle = sk.Style(iconstyle=iconstyle) # 共通の<Style>

        iter_ = self._format_times(self.times), self.lats, self.lons
        range_ = 0, None, cnf.sampling_step
        for i, (t, lat, lon) in enumerate(islice(zip(*iter_), *range_)):

            # <Placemark> <Point> <coordinates>
            pnt = kml.newpoint(coords=[(lon, lat)])
            pnt.timestamp.when = t # <TimeStamp> <when>
            pnt.style = sharedstyle
        else:
            print "sum length of kml data:", i + 1

        self._save(kml, savename, cnf.kmz)

class ActionAnimationKml(AnimationKml):
    """アクションによりアニメーションの変化するKmlを作成するクラス"""

    def __init__(self, times, lats, lons, acts, act_icons=None):
        """Constructor

        :param acts : iterable of strs
            'times'に対応するアクション名のリスト

        :param act_icons : dict of str or None
            それぞれのアクションに対応するアイコンの辞書
            {'action': 'resorce of icon'}
            Noneの場合はランダムな画像が生成される
            指定したほうが若干高速に動く

        See also AnimationKml.__init__()
        """

        super(ActionAnimationKml, self).__init__(times, lats, lons)

        # アイコン初期化
        if act_icons is None:
            self.acts = list(acts)
            act_icons = {a: drow_random_color_circle(
                (16, 16), T('{}.png'.format(a))) for a in set(self.acts)}
        else:
            self.acts = acts

        self.act_icons = act_icons

    def to_animatable(self, savename, kml_cnf):
        """Override

        See AnimationKml.to_animatable()
        """

        cnf = self._parse_conf(kml_cnf)
        kml = sk.Kml()

        # <IconStyle>
        sharedstyles = {a:sk.Style(sk.IconStyle(icon=sk.Icon(href=ic),
                                                  scale=cnf.iconscale,
                                                  hotspot=cnf.hotspot))
                        for a, ic in self.act_icons.items()}

        iter_ = self._format_times(self.times), self.lats, self.lons, self.acts
        range_ = 0, None, cnf.sampling_step
        for i, (t, lat, lon, act) in enumerate(islice(zip(*iter_), *range_)):
            # <Placemark> <Point> <coordinates>
            pnt = kml.newpoint(coords=[(lon, lat)])
            pnt.style = sharedstyles[act]
            pnt.timestamp.when = t # <TimeStamp> <when>
        else:
            print "sum length of kml data:", i + 1

        self._save(kml, savename, cnf.kmz)