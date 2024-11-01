from PIL import Image, ImageDraw, ImageFont


# 创建一个白底黑字的图片
def create_image(number):
    # 创建一个白色的背景
    img = Image.new('RGB', (100, 100), 'white')
    draw = ImageDraw.Draw(img)

    font_path = "smallfont2.ttf"  # 请根据实际路径设置
    font_size = 60  # 设置字体大小
    font = ImageFont.truetype(font_path, font_size)
    text = str(number)

    # 计算文本边界框以进行居中
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 计算文本位置以居中
    text_x = (img.width - text_width) / 2
    text_y = (img.height - text_height) / 2

    # 画出黑色文本
    draw.text((text_x, text_y), text, fill='black', font=font)

    # 保存图片
    img.save(f'E:\\Program\\15_Digital_Issues\\15digital\\Digital_Images\\{number}.png')


# 生成0到15的图片
for i in range(16):
    create_image(i)
