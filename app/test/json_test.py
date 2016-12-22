# coding: utf-8

if __name__ == '__main__':
    import json
    from app import T
    f = open(T('test.json'), 'r')
    l = json.load(f)
    f.close()
    print l