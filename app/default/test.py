# coding: utf-8

class Hello():

    def __init__(self):

        self.hello = "Hello, World!"

    def print_hello(self):
        print self.hello

    def print_hellos(self):
        for i in xrange(100):
            print i+1,
            self.print_hello()

    def print_ninenine(self):
        for i in xrange(9):
            for j in xrange(9):
                print "%d\t" % ((i+1)*(j+1)),
            print

if __name__ == '__main__':
    # Helloのオブジェクトをつくる
    hello = Hello()

    # print_helloメソッドを呼び出す
    hello.print_ninenine()