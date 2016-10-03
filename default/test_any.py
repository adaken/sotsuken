def csv_test():
    import csv
    with open('eggs.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
        spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
        snakes = [["AAAAAAAAAA", "ABABABA"], ["BBBBBBBBBB", "BABAB"]]
        spamwriter.writerows(snakes)

def array_test():
    a = [[0 for i in range(2)] for j in range(3)]
    a[0][0] = 2
    print a

def transpose_test():

    def print_array(a):
        for i in xrange(len(a)):
            for j in a[i]:
                print j,
            print

    def transpose(list2):
        row_size = len(list2)
        col_size = len(list2[0])
        nlist2 = [[None for i in xrange(row_size)] for j in xrange(col_size)]
        for i in xrange(row_size):
            for j in range(col_size):
                nlist2[j][i] = list2[i][j]
        return nlist2

    array = [[1, 2], [3, 4], [5, 6]]

    print_array(array)
    print
    print_array(transpose(array))

def strop_test():
    s = "http://www"
    print s[0:7]

def numpy_test():
    import numpy as np

if __name__ == "__main__":
    transpose_test()