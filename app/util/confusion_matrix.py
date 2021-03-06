# coding: utf-8
import numpy as np

class ConfusionMatrix(object):
    """コンフュージョン・マトリクス"""

    def __init__(self, n):
        """コンストラクタ"""
        self.sum_matrix = np.zeros([n]*2, dtype=np.float64) # 零行列で初期化

    def __call__(self, conf_matrix):
        """update()へのアクセサ"""
        return self.update(conf_matrix)

    def update(self, conf_matrix):
        """混同行列を更新"""
        self.matrix = conf_matrix # 実引数で混同行列を更新
        self.sum_matrix += self.matrix
        self.tp = np.diag(self.matrix) # TP(対角項)
        return self

    @property
    def precision(self):
        """適合率"""
        total = self.matrix.sum(axis=1)
        prec = self.tp / total.astype(np.float64)
        prec[np.isnan(prec)] = 1.
        return prec

    @property
    def recall(self):
        """再現率"""
        total = self.matrix.sum(axis=0)
        rec = self.tp / total.astype(np.float64)
        rec[np.isnan(rec)] = 1.
        return rec

    @property
    def fmeasure(self):
        """F値"""
        prec, rec = self.precision, self.recall
        rp = rec + prec
        fm = (2 * rec * prec) / rp
        fm[np.isnan(fm)] = 0.
        return fm

if __name__ == '__main__':
    l = [[20, 0, 0],
         [5, 15, 0],
         [0, 0, 20]]

    a = np.array(l)
    conf = ConfusionMatrix(3)
    conf(a)
    print conf.precision
    print conf.recall
    print conf.fmeasure
