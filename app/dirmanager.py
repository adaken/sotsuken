# coding: utf-8

import os
import glob
import re
import warnings

class Dir(object):
    """簡単なディレクトリ操作"""

    parsing = False # パス解析中かどうか
    permit_empty_file = False # 存在しないファイル名指定を許容するかどうか
    auto_mkdir = False # 存在しないディレクトリを指定した場合に作成するかどうか

    def __init__(self, root, emptyfile=True):
        os.chdir(os.path.split(__file__)[0]) # このモジュールのディレクトリにcd
        assert os.path.exists(root), "No such directory"
        self.path = os.path.abspath(root) # 引数の絶対パス
        Dir.permit_empty_file = emptyfile
        self.updata()

    def __call__(self, *key, **kwargs):
        """インスタンスを()で呼び出すとget()にアクセスできる"""

        return self.get(*key, **kwargs)

    def updata(self):
        """索引を更新"""

        # ディレクトリ下のファイルの絶対パスのリストを入手
        paths = glob.glob(self.path + '\\*')
        # サブディレクトリ
        self.subdirs = [Dir(p) for p in paths if os.path.isdir(p)]
        # サブディレクトリ名リスト
        self.subdir_names = [n.name for n in self.subdirs]
        # ファイル名リスト
        self.filenames = [os.path.basename(p)
                          for p in paths if os.path.isfile(p)]

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

    def _get_file_or_dir(self, name, mkdir=False, updated=False):
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
            #print "parsing?:", Dir.parsing
            #print "automkdir?:", Dir.auto_mkdir
            #print "permit_empty_file?", Dir.permit_empty_file
            if Dir.parsing and Dir.auto_mkdir: # パス解析中かつ自動作成ON
                self.mkdir(name)
                return self._getdir(name)
            # パス解析中でないかつ存在しないファイル名もOKである
            elif (not Dir.parsing) and Dir.permit_empty_file:
                return self.path + '\\' + name
            else:
                raise ValueError(u"no such file or directory: '{}'"
                                 .format(name))
        elif not updated:
            # 新しいファイルが作成された可能性があるため、索引を更新
            self.updata()
            return self._get_file_or_dir(name, mkdir, updated=True)
        else:
            raise RuntimeError

    def _getr(self, path, mkdir):
        """再帰的にパスを解析"""

        Dir.auto_mkdir = mkdir
        Dir.parsing = True
        # パスを分割
        names = iter(filter(lambda w: len(w) > 0, self._splitp(path)))
        def f(d=self, n=names.next()): # 再帰関数
            try:
                n_ = names.next()
            except StopIteration:
                Dir.parsing = False
                Dir.auto_mkdir = False
            except:
                Dir.parsing = False
                Dir.auto_mkdir = False
            p = d.get(n)
            if not Dir.parsing: # 解析終了
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

        return self.path + '\\' + name

    def _getdir(self, name):
        """名前からDirオブジェクト"""

        i = self.subdir_names.index(name)
        return self.subdirs[i]

    @property
    def name(self):
        """このディレクトリの名前"""

        return os.path.basename(self.path)

    def ls(self, abs=False):
        """lsコマンドみたいなリスト

        :param abs : bool, default: False
            True : 絶対パスのリスト
            False: 名前のみのリスト

        :return dirs_and_files: list of str
            [[dirs], [files]]
        """

        if abs:
            ret = [map(self._getabs, self.subdir_names),
                   map(self._getabs, self.filenames)]
        else:
            ret = [self.subdir_names, self.filenames]
        return ret

    def mkdir(self, name):
        """配下にディレクトリを作成"""

        if not self._exists(name):
            os.mkdir(self._getabs(name))
            self.updata()
        else:
            warnings.warn(u"directory already exists: {}".foamat(name))
        return self._getabs(name)

    def rm(self, name):
        """配下のファイルを削除"""

        if self._exists(name):
            if not os.path.isfile(self._getabs(name)):
                raise ValueError(u"'name' must be file name, " \
                                 u"not directory name: {}".format(name))
            os.remove(self._getabs(name))
            self.updata()
        else:
            warnings.warn("Already not exists: {}".format(name))

    def rmdir(self, name):
        """配下のディレクトリを削除"""

        if self._exists(name):
            if not os.path.isdir(self._getabs(name)):
                raise ValueError(u"'name' must be directory name, " \
                                 u"not file name: {}".format(name))
            os.removedirs(self._getabs(name))
            self.updata()
        else:
            warnings.warn(u"already non-exists: {}".format(name))

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
    c = Res()
    print a is b
    print a is c
    print a.ls(True)
    print list(a('bigfile', 'data'))
    print a('data')('acc')()
    print a