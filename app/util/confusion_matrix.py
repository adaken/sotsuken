# coding: utf-8

class ConfusionMatrix(object):
    """コンフュージョン・マトリクス"""

    def __init__(self, n):
        self.sum_matrix = np.zeros([n]*2, dtype=np.int32)

    def __call__(self, conf_matrix):
        self.matrix = conf_matrix
        self.sum_matrix += self.matrix
        self.tp = np.diag(self.matrix) # TP(対角項)

    @property
    def precision(self):
        """適合率"""
        total = self.matrix.sum(axis=1)
        prec = self.tp / total.astype(np.float32)
        prec[np.isnan(prec)] = 1
        return prec

    @property
    def recall(self):
        """再現率"""
        total = self.matrix.sum(axis=0)
        rec = self.tp / total.astype(np.float32)
        rec[np.isnan(rec)] = 1
        return rec

    @property
    def fmeasure(self):
        """F値"""
        prec, rec = self.precision, self.recall
        fm = (2 * rec * prec) / (rec + prec)
        fm[np.isnan(fm)] = 1
        return fm

if __name__ == '__main__':
    l = [[20, 0, 0],
         [5, 15, 0],
         [0, 0, 20]]

    import numpy as np
    a = np.array(l)
    conf = ConfusionMatrix(3)
    conf(a)
    print conf.precision
    print conf.recall
    print conf.fmeasure
