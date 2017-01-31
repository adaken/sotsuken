# coding: utf-8

if __name__ == '__main__':
    from app.util import save_gps_as_json
    from app import R, L
    save_gps_as_json(xlsx=R('data/raw/'))