# coding: utf-8

if __name__ == '__main__':
    from util.fft import fft
    from util.util import make_input_from_xlsx
    from util.excelwrapper import ExcelWrapper

    filename, sheetname = r'E:\work\data\run.xlsx', 'Sheet4'
    for i in range(1):
        vecs = ExcelWrapper(filename, sheetname).select_column(column_letter='F', begin_row=50, end_row=50+127, log=True)
        fftmag, fig = fft(vecs, fft_points=128, out_fig=True)
        fig.savefig(r'E:\work\fig\fft_test_fig.png')
    print "finish"

    """
    inputv = make_input_from_xlsx(filename='E:\work\new_run.xlsx', sheetname='Sheet4', col='F',
                                  read_range=(2, None), sampling='rand', sample_cnt=100, overlap=0,
                                  fft_N=128, normalizing='01', label='run', log=True)
    """