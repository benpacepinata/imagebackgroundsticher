from PIL import Image


class AlphaComposite:
    def __call__(self, im_rgb, bg_img):
        bg_img = bg_img.convert("RGBA")
        im_rgb = im_rgb.convert("RGBA")
        result = Image.alpha_composite(bg_img, im_rgb)
        return result


