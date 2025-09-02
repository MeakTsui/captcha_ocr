from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw, ImageFont
import random
import string
import os

def random_text(length=6):
    chars = string.ascii_uppercase
    return ''.join(random.choices(chars, k=length))

def add_noise_lines(draw, width, height, line_count=10):
    for _ in range(line_count):
        start = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        color = tuple(random.randint(100, 200) for _ in range(3))
        draw.line([start, end], fill=color, width=2)

def add_background_text(draw, width, height, font):
    bg_text = random_text(6)
    font_size = 36
    for i in range(2):
        x = random.randint(0, width-100)
        y = random.randint(0, height-40)
        color = (180, 180, 180, 80)
        draw.text((x, y), bg_text, font=font, fill=color)

# 获取 captcha 库自带字体路径
import captcha
font_path = os.path.join(os.path.dirname(captcha.__file__), 'data', 'fonts', 'DejaVuSans.ttf')

width, height = 200, 70
captcha_text = random_text(6)
image_captcha = ImageCaptcha(width=width, height=height, fonts=[font_path])
captcha_image = image_captcha.generate_image(captcha_text)

draw = ImageDraw.Draw(captcha_image)
font = ImageFont.truetype(font_path, 36)

add_noise_lines(draw, width, height, line_count=10)
add_background_text(draw, width, height, font)

captcha_image.save('captcha_with_noise.png')
print(f"验证码内容：{captcha_text}，图片已保存为 captcha_with_noise.png")