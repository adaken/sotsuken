# coding: utf-8

import simplekml as simk

class KmlWrapper:

    def __init__(self):
        pass

    def createAnimeKml(self,
                       save_path,
                       times,
                       longitudes,
                       latitudes,
                       format_time = True,
                       sampling_interval = 0, # 0 to no interval
                       icon_res = 'http://maps.google.com/mapfiles/kml/shapes/man.png',
                       icon_scale = 1):
        """
        時間、経度、緯度の配列からアニメーション可能なkmlを生成します
        """

        # リストのサイズが同じかチェック
        if not times.size == longitudes.size == latitudes.size:
            print "Error: times, longitude and latitudes must be same size ndarray"
            return

        # リソースが利用可能かチェック
        if not self.__available_resorce(icon_res): return

        # タイムスタンプの形式を変更
        if format_time:
            self.__format_time(times)

        # <IconStyle>
        icon = simk.Icon(href=icon_res)
        hotspot = simk.HotSpot(x=0, xunits=simk.Units.fraction,
                               y=0, yunits=simk.Units.fraction)
        iconstyle = simk.IconStyle(icon=icon, scale=icon_scale, hotspot=hotspot)

        # 共通の<Style>
        sharedstyle = simk.Style(iconstyle=iconstyle)

        # kml生成
        kml = simk.Kml()
        for i in xrange(0, times.size, sampling_interval + 1):

            # <Placemark> <Point> <coordinates>
            pnt = kml.newpoint(coords=[(longitudes[i], latitudes[i])])
            pnt.style = sharedstyle

            # <TimeStamp> <when>
            pnt.timestamp.when = times[i]

        # ファイルに出力
        kml.save(save_path, True)

    def __available_resorce(self, res):
        """
        ファイルの存在とかをチェック
        """

        # URLなら何もしない
        if len(res) >= 7:
            if res[0:7] == "http://" or res[0:8] == "https://":
                return True

        # URL以外ならチェック
        import os
        if os.path.exists(res):
            return True
        else:
            print "Error: input resorce is not exist: '%s'" % res
            return False

    def __format_time(self, times):
        """
        タイムスタンプの形式を整える
        yyyy/m/d hh:mm:ss.mmm -> yyyy-mm-dd"T"hh:mm:ss.mmm"Z"
        """

        from datetime import datetime
        format_before = "%Y-%m-%d %H:%M:%S.%f" # 元の形式
        format_before_zero = "%Y-%m-%d %H:%M:%S" # microsecondが0のときの形式
        format_after = "%Y-%m-%dT%H:%M:%S.%fZ" # 新しい形式
        for i in xrange(times.size):
            old_time = None

            try:
                old_time = datetime.strptime(times[i], format_before)
            except ValueError:
                old_time = datetime.strptime(times[i], format_before_zero)

            times[i] = old_time.strftime(format_after)
            #print "change timestamp format: %s -> %s" % (old_time, times[i])

if __name__ == "__main__":
    pass