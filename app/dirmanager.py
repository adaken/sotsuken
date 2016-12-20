# coding: utf-8

import os
import glob
import re
import warnings

class Dir(object):
    """簡単なディレクトリ操作"""

    parsing = False # パス解析中かどうか
    permit_empty_file = False # 存在しないファイル名指定を許容するかどうか
    auto_mkdir = False # 存在しないにディレクトリを指定した場合に作成するかどうか

    def __init__(self, root, emptyfile=True):
        os.chdir(os.path.split(__file__)[0]) # このディレクトリにcd
        assert os.path.exists(root)
        self.path = os.path.abspath(root) # 引数の絶対パス
        Dir.permit_empty_file = emptyfile
        self.updata()

    def __call__(self, *key, **kwargs):
        """インスタンスを()で呼び出すとgetにアクセスできる"""

        return self.get(*key, **kwargs)

    def updata(self):
        paths = glob.glob(self.path + '\\*') # ディレクトリ下の絶対パスのリストを入手
        self.subdirs = [Dir(p) for p in paths if os.path.isdir(p)] # サブディレクトリ
        self.subdir_names = [n.name for n in self.subdirs] # サブディレクトリ名リスト
        # ファイル名リスト
        self.filenames = [os.path.basename(p) for p in paths if os.path.isfile(p)]

    def get(self, *args, **kwargs):
        """ディレクトリ配下の絶対パスかDirオブジェクトを入手

        :param name : *args
            指定しないかディレクトリ名かファイル名か相対パス

        :return abs_path or dir or iter
            空 -> 自分自身の絶対パス
            ファイル名 -> 絶対パス
            ディレクトリ名 -> Dirオブジェクト
            相対パス -> 絶対パスかDirオブジェクト
        """

        len_ = len(args)
        if len_ == 0:
            return self.path
        elif len_ == 1:
            return self._get_file_or_dir(args[0], **kwargs)
        else:
            def _gen(*keys):
                for key in keys:
                    yield self._get_file_or_dir(key, **kwargs)
            return _gen(*args)

    def _get_file_or_dir(self, name, mkdir=False):
        """名前からファイルの絶対パスかDirオブジェクト"""

        isfile = lambda a: a in self.filenames
        isdir = lambda a: a in self.subdir_names

        if self._ispath(name):       # パスである
            return self._getr(name, mkdir)
        elif isfile(name):           # ファイルリストにある
            return self._getabs(name)
        elif isdir(name):            # ディレクトリリストにある
            return self._getdir(name)
        elif not self._exists(name): # ファイルシステムに存在しない
            print "parsing?:", Dir.parsing
            print "automkdir?:", Dir.auto_mkdir
            if Dir.parsing and Dir.auto_mkdir: # パス解析中かつ自動作成ON
                self.mkdir(name)
                return self._getdir(name)
            # パス解析中でないかつ存在しないファイル名もOKである
            elif (not Dir.parsing) and Dir.permit_empty_file:
                return self.path + '\\' + name
            else:
                raise ValueError("No such file or directory: '{}'".format(name))
        else:
            raise RuntimeError

    def _getr(self, path, mkdir):
        """再帰的にパスを解析"""

        Dir.auto_mkdir = mkdir
        Dir.parsing = True
        names = iter(filter(lambda w: len(w) > 0, self._splitp(path)))
        def f(d=self, n=names.next()):
            try:
                n_ = names.next()
            except StopIteration:
                Dir.parsing = False
                Dir.auto_mkdir = False
            except:
                Dir.parsing = False
            p = d.get(n)
            if not Dir.parsing: # 解析終了
                return p
            return f(p, n_)
        return f()

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

        return self.path + '\\' + name

    def _getdir(self, name):
        """名前からDirオブジェクト"""

        i = self.subdir_names.index(name)
        return self.subdirs[i]

    @property
    def name(self):
        """このディレクトリの名前"""

        return os.path.basename(self.path)

    @property
    def ls(self):
        """lsコマンドみたいなリスト

        :return dirs_and_files: list of str
            [[dirs], [files]]

        """
        return [self.subdir_names, self.filenames]

    def mkdir(self, name):
        """配下にディレクトリを作成"""

        if not self._exists(name):
            os.mkdir(self._getabs(name))
            self.updata()
        else:
            warnings.warn("directory already exists")
        return self._getabs(name)

    def rmdir(self, name):
        """配下のディレクトリを削除"""

        if self.exits(name):
            os.removedirs(self._getabs(name))
            self.updata()
        else:
            warnings.warn("directory not exists")

    def _exists(self, name):
        return os.path.exists(self._getabs(name))

class Log(Dir):
    """ログディレクトリsingleton"""

    _instance = None

    def __init__(self):
        """書いちゃダメ"""
        pass

    def __new__(cls, root='..\log', **kwargs):
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

    def __new__(cls, root=r'..\res', **kwargs):
        if cls._instance is None:
            """初期化処理"""
            ins = cls._instance = object.__new__(cls)
            super(Res, ins).__init__(root, **kwargs)
            """
            ins.bigfile = ins('bigfile')
            ins.data = ins('data')
            ins.img = ins('img')
            ins.misc = ins('misc')
            ins.sound = ins('sound')
            """
        return cls._instance

class Tmp(Dir):
    """Tmpディレクトリsingleton"""

    _instance = None

    def __init__(self):
        """書いちゃダメ"""
        pass

    def __new__(cls, root=r'..\tmp', **kwargs):
        if cls._instance is None:
            ins = cls._instance = object.__new__(cls)
            super(Tmp, ins).__init__(root, **kwargs)
        return cls._instance

L = Log()
R = Res()
T = Tmp()

if __name__ == '__main__':
    a = Res()
    b = Tmp()
    print a is b
    print a.ls
    print list(a('bigfile', 'data'))
    print a('data')('acc')()
    print a('data/gps/af/log.txt', mkdir=False)
