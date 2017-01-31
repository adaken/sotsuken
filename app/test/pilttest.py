# coding: utf-8

if __name__ == '__main__':

    def drow_circle(rgb, size, savepath):
        assert isinstance(rgb, tuple)
        assert isinstance(size, tuple)
        from PIL import Image
        from PIL import ImageDraw
        im= Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im)
        draw.ellipse(((1, 1), size), outline=None, fill=rgb)
        del draw
        im.save(savepath)