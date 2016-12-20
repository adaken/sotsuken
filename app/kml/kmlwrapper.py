# coding: utf-8

import simplekml as simk

class KmlWrapper:

    def __init__(self):
        pass

    def createAnimeKml(self,
                       save_path,
                       times,
                       lons,
                       lats,
                       acts=None,
                       act_icons=None,
                       icon_res='http://maps.google.com/mapfiles/kml/shapes/man.png',
                       icon_scale=1,
                       format_time=True,
                       sampling_step=1):
        """
        時間、経度、緯度、アクションのリストからアニメーション可能なkmlを生成します

        Parameters
        ----------
        save_path : str
            kmlを保存するパス

        times, lons, lats, acts : list or tuple
            これらはすべて同じサイズのリストでなければならない。
            time : strs
                時間のリスト
            lons : floats
                経度のリスト
            lats : floats
                緯度のリスト
            acts : strs, default: None
                その時間におけるアクション名のリスト

        act_icons : dic, default:None
            アクション名を指定した際に使われる、アクション名をkeyとするアイコンリソースの辞書
            {str(アクション名) : str(アクション名に対応するアイコンリソース)}

        icon_res : str, default:'http://maps.google.com/mapfiles/kml/shapes/man.png':
            アクションを指定しない場合のアイコンリソース

        icon_scale : int, default: 1
            アイコンの大きさ

        format_time : bool, default: True
            時間のフォーマットを変更するかどうか

        sampling_step : int, default: 1
            step > 0
            読み込みペース
            指定した分だけ次の読み込み行番号を増やす
        """

        # アクションの指定、リストのサイズが同じかをチェック
        if not (acts is None and act_icons is None):
            print "アクション指定あり"
            if acts is None:
                raise AssertionError("act_iconsを指定した場合、actsも指定する必要があります。")
            elif act_icons is None:
                raise AssertionError("actsを指定した場合、act_iconsも指定する必要があります。")
            if not len(times) == len(lons) == len(lats) == len(acts):
                raise ValueError("""引数times, lons, lats, actsのサイズが違います。
これらのサイズをそろえる必要があります。: times: {}, lons: {}, lats: {}, acts: {}"""
                .format(len(times), len(lons), len(lats), len(acts)))
        else:
            print "アクション指定なし"
            assert len(times) == len(lons) == len(lats), "引数times, lons, latsのサイズが違います。これらのサイズをそろえる必要があります。"

        # リソースが利用可能かチェック
        if not self._is_resorce_available(icon_res):
            raise AssertionError("リソースが存在しません: {}".format(icon_res))

        # タイムスタンプの形式を変更
        if format_time:
            print "タイムスタンプの形式を変更中です..."
            times = self._format_times(times)

        print "kmlを作成中です..."
        kml = simk.Kml()
        hotspot = simk.HotSpot(x=0, xunits=simk.Units.fraction,
                               y=0, yunits=simk.Units.fraction)

        # アクションを指定しなかった場合
        if acts is None:

            # <IconStyle>
            icon = simk.Icon(href=icon_res)
            iconstyle = simk.IconStyle(icon=icon, scale=icon_scale, hotspot=hotspot)

            # 共通の<Style>
            sharedstyle = simk.Style(iconstyle=iconstyle)

            for i in xrange(0, len(times), sampling_step):

                # <Placemark> <Point> <coordinates>
                pnt = kml.newpoint(coords=[(lons[i], lats[i])])
                pnt.style = sharedstyle

                # <TimeStamp> <when>
                pnt.timestamp.when = times[i]

        # アクションを指定した場合
        else:

            # <IconStyle>
            sharedstyles = {act:simk.Style(simk.IconStyle(icon=simk.Icon(href=icon),
                                                          scale=icon_scale, hotspot=hotspot))
                            for act, icon in act_icons.items()}

            for i in xrange(0, len(times), sampling_step):

                # <Placemark> <Point> <coordinates>
                pnt = kml.newpoint(coords=[(lons[i], lats[i])])
                pnt.style = sharedstyles[acts[i]]

                # <TimeStamp> <when>
                pnt.timestamp.when = times[i]

        # ファイルに出力
        print "kmlを保存中です..."
        kml.save(save_path, True)
        print "保存が完了しました"

    def _is_resorce_available(self, res):
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
            return False

    def _format_times(self, times):
        """
        タイムスタンプの形式を整える
        yyyy/m/dirmanager hh:mm:ss.mmm -> yyyy-mm-dd"T"hh:mm:ss.mmm"Z"
        "%Y-%m-%dirmanager %H:%M:%S.%f" # 元の形式
        "%Y-%m-%dirmanager %H:%M:%S" # microsecondが0のときの形式
        """
        format_new = "%Y-%m-%dT%H:%M:%S.%fZ" # 新しい形式
        return [t.strftime(format_new) for t in times]

if __name__ == "__main__":
    pass