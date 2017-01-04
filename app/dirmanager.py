# coding: utf-8

import os
import glob
import re
import warnings
import shutil

class Dir(object):
    """ディレクトリを表すクラス"""


    """クラス変数"""
    permit_empty_file = True # 存在しないファイル名指定を許容するかどうか
    _auto_mkdir = False # 存在しないディレクトリを指定した場合に作成するかどうか
    _parsing = False # パス解析中かどうか

    def __init__(self, root):
        os.chdir(os.path.split(__file__)[0]) # このモジュールのディレクトリにcd
        if not os.path.exists(root):
            raise ValueError(u"no such directory")
        self.p = os.path.abspath(root) # このインスタンスが表すディレクトリ

    def __call__(self, *key, **kwargs):
        """get()へのアクセサ

        インスタンスを()で呼び出すとget()にアクセスできる
        """

        return self.get(*key, **kwargs)

    def __str__(self):
        return "A Dir object represents '{}'".format(self.p)

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        """このディレクトリの名前"""

        return os.path.basename(self.p)

    @property
    def subdirs(self):
        """サブディレクトリのDirオブジェクトのリスト"""

        return [Dir(p) for p in self._lowers if os.path.isdir(p)]

    @property
    def subdirnames(self):
        """サブディレクトリの名前のリスト"""

        return [os.path.basename(p) for p in self._lowers if os.path.isdir(p)]

    @property
    def filenames(self):
        """配下のファイルの名前のリスト"""

        return [os.path.basename(p)
                for p in self._lowers if os.path.isfile(p)]

    @property
    def _lowers(self):
        """配下のファイル、ディレクトリの絶対パスのリスト"""

        return glob.glob(self.p + '\\*')

    def get(self, *args, **kwargs):
        """ディレクトリ配下の絶対パスかDirオブジェクトを入手

        :param name : *args
            指定しないかディレクトリ名かファイル名か相対パス

        :param mkdir : bool, default: False, optional
            True: 'name'にパスを指定した場合に足りないディレクトリを作成

        :return abs_path or dir or iter
            空 -> 自分自身の絶対パス
            ファイル名 -> 絶対パス
            ディレクトリ名 -> Dirオブジェクト
            パス -> 絶対パスかDirオブジェクト
        """

        len_ = len(args)
        if len_ == 0: # 空
            return self.p
        elif len_ == 1: # 引数が1つ
            return self._get_file_or_dir(args[0], **kwargs)
        else: # 引数が複数
            def _gen(*keys):
                for key in keys:
                    yield self._get_file_or_dir(key, **kwargs)
            return _gen(*args)

    def ls(self, absp=False, ext=True):
        """lsコマンドみたいなの

        :param abs : bool, default: False
            True : 絶対パスのリスト
            False: 名前のみのリスト

        :param ext : bool, default: True
            True: 拡張子あり
            False: 拡張子なし

        :return dirs_and_files: list of str
            [[dirs], [files]]
        """

        lowers = self._lowers
        subdirs = [l for l in lowers if os.path.isdir(l)]
        files = [l for l in lowers if os.path.isfile(l)]

        if not ext:
            subdirs = [os.path.splitext(s)[0] for s in subdirs]
            files = [os.path.splitext(f)[0] for f in files]
        if not absp:
            subdirs = [os.path.basename(s) for s in subdirs]
            files = [os.path.basename(f) for f in files]

        return [subdirs, files]

    def mkdir(self, name):
        """配下にディレクトリを作成"""

        if not self._exists(name):
            os.mkdir(self._getabs(name))
        else:
            warnings.warn(u"directory already exists: {}".format(name))
        return self._getabs(name)

    def rm(self, name):
        """配下のファイルを削除"""

        if self._exists(name):
            if not os.path.isfile(self._getabs(name)):
                raise ValueError(u"'name' must be file name, " \
                                 u"not directory name: {}".format(name))
            os.remove(self._getabs(name))
        else:
            warnings.warn("Already not exists: {}".format(name))

    def rmdir(self, name, ignore=False):
        """配下のディレクトリを削除"""

        if self._exists(name):
            if not os.path.isdir(self._getabs(name)):
                raise ValueError(u"'name' must be directory name, " \
                                 u"not file name: {}".format(name))
            if ignore:
                shutil.rmtree(self._getabs(name))
            else:
                os.rmdir(self._getabs(name))
        else:
            warnings.warn(u"already non-exists: {}".format(name))

    def _get_file_or_dir(self, name, mkdir=False, isretry=False):
        """名前からファイルの絶対パスかDirオブジェクト"""

        if self._ispath(name):         # パスである
            return self._getr(name, mkdir)
        elif name in self.filenames:   # ファイルリストにある
            return self._getabs(name)
        elif name in self.subdirnames: # ディレクトリリストにある
            return self._getdir(name)
        elif not self._exists(name):   # ファイルシステムに存在しない
            #print "parsing?:", Dir.parsing
            #print "automkdir?:", Dir._auto_mkdir
            #print "permit_empty_file?", Dir.permit_empty_file
            if Dir._parsing and Dir._auto_mkdir: # パス解析中かつ自動作成ON
                self.mkdir(name)
                return self._getdir(name)
            # パス解析中でないかつ存在しないファイル名もOKである
            elif (not Dir._parsing) and Dir.permit_empty_file:
                return self._getabs(name)
            else:
                raise ValueError(u"no such file or directory: '{}'"
                                 .format(name))
        elif not isretry:
            return self._get_file_or_dir(name, mkdir, isretry=True) # リトライ
        else:
            raise RuntimeError

    def _getr(self, path, mkdir):
        """再帰的にパスを解析"""

        Dir._auto_mkdir = mkdir
        Dir._parsing = True
        # パスを分割
        names = iter(filter(lambda w: len(w) > 0, self._splitp(path)))
        def f(d=self, n=names.next()): # 再帰関数
            try:
                n_ = names.next()
            except StopIteration:
                Dir._parsing = False
                Dir._auto_mkdir = False
            p = d.get(n)
            if not Dir._parsing: # 解析終了
                return p
            return f(p, n_)
        return f() # 再帰実行

    def _ispath(self, path):
        """パスかどうか判別"""

        if re.search(r'[\\/]+', path):
            return True
        return False

    def _splitp(self, path):
        """パスを名前のリストに分割"""

        return re.split(r'[\\/]+', path)

    def _getabs(self, name):
        """名前からファイルの絶対パス"""

        return self.p + '\\' + name

    def _getdir(self, name):
        """名前からDirオブジェクト"""

        return Dir(self._getabs(name))

    def _exists(self, name):
        return os.path.exists(self._getabs(name))

class Log(Dir):
    """ログディレクトリsingleton"""

    _instance = None

    def __init__(self):
        """書いちゃダメ"""
        pass

    def __new__(cls, root='../log', **kwargs):
        if cls._instance is None:
            ins = cls._instance = object.__new__(cls)
            super(Log, ins).__init__(root, **kwargs)
        return cls._instance

class Res(Dir):
    """リソースディレクトリsingleton"""

    _instance = None

    def __init__(self):
        """書いちゃダメ"""
        pass

    def __new__(cls, root='../res', **kwargs):
        if cls._instance is None:
            ins = cls._instance = object.__new__(cls)
            super(Res, ins).__init__(root, **kwargs)
        return cls._instance

class Tmp(Dir):
    """Tmpディレクトリsingleton"""

    _instance = None

    def __init__(self):
        """書いちゃダメ"""
        pass

    def __new__(cls, root='../tmp', **kwargs):
        if cls._instance is None:
            ins = cls._instance = object.__new__(cls)
            super(Tmp, ins).__init__(root, **kwargs)
        return cls._instance

L = Log() # ログ用ディレクトリ
R = Res() # リソースディレクトリ
T = Tmp() # 一時ファイル用ディレクトリ

if __name__ == '__main__':
    def test1():

        # singletonか確認
        dir_ = Dir('../res')
        print "Dir is not singleton?:", dir_ is not Dir('../res')
        log = Log()
        print "Log is singleton?:", log is Log()
        res = Res()
        print "Res is singleton?:", res is Res()
        tmp = Tmp()
        print "tmp is singleton?:", tmp is Tmp()

    def test2():

        # 無駄に__init__()が呼ばれないか
        res = Res()
        fns = res.filenames
        sns = res.subdirnames
        a = res('data/gps/player')

    def test3():
        r = Res() # Dirクラスを継承したResオブジェクト

        # プロパティの確認
        print "directory name:", r.name
        print "directory path:", r.p
        print "names of files and dirs:", r.ls()
        print "paths of files and dirs:", r.ls(absp=True)
        print "subdir names:", r.subdirnames
        print "file names  :", r.filenames

        # ルートパスの取得
        print "self path:", r()
        print "self path:", r.p

        # 配下のDirオブジェクトを取得
        print "subdir object:", r.get('data')
        print "subdir object:", r('data')

        # 配下のディレクトリのパスを取得
        print "subdir path  :", r('data').p
        print "subdir path  :", r('data')()
        print "subdir path  :", r.get('data').p
        print "subdir path  :", r('\/data/').p

        # 2階層以上下のパスを取得
        print "subdir path2 :", r.get('data/gps/players').p
        print "subdir path2 :", r('data/gps')('players').p
        print "subdir path2 :", r.get('data').get('gps').p
        print "subdir path2 :", r('data')('gps')()

        # ディレクトリ操作
        # 存在しないファイル名を許可しない
        Dir.permit_empty_file = False
        # 以下は存在しないファイル名なのでエラー
        #print L('hoge/fuga/test/test.txt', mkdir=True) # ディレクトリがない場合は作成

        # 許可する
        Dir.permit_empty_file = True
        # 以下はエラーがでない
        print L('hoge/fuga/test/test.txt', mkdir=True) # ディレクトリがない場合は作成

        # ディレクトリ作成
        L.mkdir('hogefuga')

        # ディレクトリ削除
        print "before removing:", L.subdirnames
        L.rmdir('hogefuga')
        #L('hoge').rmdir('fuga') # ファイルかディレクトリがあるとエラー
        L.rmdir('hoge', True) # ファイル、ディレクトリがあっても削除
        print "after removing:",  L.subdirnames

    def ls_test():
        r = Res()('data/gps')
        r.ls()
        print r.ls(True, True)
        print r.ls(False, False)
        print r.ls(True, False)
        print r.ls(False, True)

    test3()
    ls_test()