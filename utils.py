# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
from keras.preprocessing import image
import cv2
from PIL import Image, ImageFont, ImageDraw
import albumentations
import os

provinces = ['กรุงเทพมหานคร.', 'เชียงราย.', 'เชียงใหม่.', 'น่าน.', 'พะเยา.', 'แพร่.',
             'แม่ฮ่องสอน.', 'ลำปาง.', 'ลำพูน.', 'อุตรดิตถ์.', 'กาฬสินธุ์.',
             'ขอนแก่น.', 'ชัยภูมิ.', 'นครพนม.', 'นครราชสีมา.', 'บึงกาฬ.',
             'บุรีรัมย์.', 'มหาสารคาม.', 'มุกดาหาร.', 'ยโสธร.', 'ร้อยเอ็ด.',
             'เลย.', 'สกลนคร.', 'สุรินทร์.', 'ศรีสะเกษ.', 'หนองคาย.',
             'หนองบัวลำภู.', 'อุดรธานี.', 'อุบลราชธานี.', 'อำนาจเจริญ.',
             'กำแพงเพชร.', 'ชัยนาท.', 'นครนายก.', 'นครปฐม.', 'นครสวรรค์.',
             'นนทบุรี.', 'ปทุมธานี.', 'พระนครศรีอยุธยา.', 'พิจิตร.', 'พิษณุโลก.',
             'เพชรบูรณ์.', 'ลพบุรี.', 'สมุทรปราการ.', 'สมุทรสงคราม.', 'สมุทรสาคร.',
             'สิงห์บุรี.', 'สุโขทัย.', 'สุพรรณบุรี.', 'สระบุรี.', 'อ่างทอง.',
             'อุทัยธานี.', 'จันทบุรี.', 'ฉะเชิงเทรา.', 'ชลบุรี.', 'ตราด.',
             'ปราจีนบุรี.', 'ระยอง.', 'สระแก้ว.', 'กาญจนบุรี.', 'ตาก.',
             'ประจวบคีรีขันธ์.', 'เพชรบุรี.', 'ราชบุรี.', 'กระบี่.', 'ชุมพร.',
             'ตรัง.', 'นครศรีธรรมราช.', 'นราธิวาส.', 'ปัตตานี.', 'พังงา.',
             'พัทลุง.', 'ภูเก็ต.', 'ระนอง.', 'สตูล.', 'สงขลา.',
             'สุราษฎร์ธานี.', 'ยะลา']
offsets = [45, 85, 85, 110, 100, 110,
           70, 100, 105, 85, 85,
           85, 100, 85, 70, 100,
           100, 65, 70, 85, 85,
           110, 85, 90, 85, 85,
           65, 85, 65, 65,
           65, 100, 85, 80, 85,
           100, 100, 45, 105, 85,
           85, 100, 60, 60, 65,
           95, 95, 80, 95, 95,
           90, 95, 85, 100, 100,
           90, 100, 90, 90, 110,
           50, 85, 90, 100, 100, 100,
           110, 45, 75, 85, 100,
           100, 100, 100, 110, 100,
           60, 100
           ]

# Graphich Plate BG
bg = cv2.imread('misc/scoop-4-1.jpg')
bg_ = []
for ymin in [5, 147, 298, 450, 601]:
    bg_.append(cv2.resize(bg[ymin:ymin + 86, 1:201], (270, 120)))
    bg_.append(cv2.resize(bg[ymin:ymin + 86, 207:405], (270, 120)))
    bg_.append(cv2.resize(bg[ymin:ymin + 86, 1:201], (270, 120))[..., ::-1])
    bg_.append(cv2.resize(bg[ymin:ymin + 86, 207:405], (270, 120))[..., ::-1])

fontpath = "./fonts/Sarun_ThangLuang.ttf"
fontpaths = ["./fonts/Sarun_ThangLuang.ttf", "./fonts/THSarabun.ttf", "./fonts/basis.ttf", "./fonts/CSChatThai.ttf"]
font = ImageFont.truetype(fontpath, 80)
font2 = ImageFont.truetype(fontpath, 35)
fonts = [ImageFont.truetype(f, 80) for f in fontpaths]
fonts2 = [ImageFont.truetype(f, 35) for f in fontpaths]
upper_offsets = [{7: 15, 6: 30, 5: 45, 4: 60, -1: 75, 0: -60, 'p': 40},
                 {7: 35, 6: 50, 5: 65, 4: 80, -1: 95, 0: -20, 'p': 60},
                 {7: 45, 6: 60, 5: 75, 4: 90, -1: 105, 0: -20, 'p': 60},
                 {7: 45, 6: 60, 5: 75, 4: 90, -1: 105, 0: -20, 'p': 60},

                 ]

# Augmented real world BG
abg = {}
abg_list = os.listdir('/media/palm/data/openimage/vrp')
while len(abg) < 100:
    path = np.random.choice(abg_list)
    abg[path] = cv2.imread(os.path.join('/media/palm/data/openimage/vrp', path))
aname = list(abg)


def random_bg():
    if np.random.randn() > 0.50:
        content = bg_[np.random.randint(0, len(bg_))]
        mil = 0
    else:
        content = np.zeros((120, 270, 3), dtype='uint8') + np.random.randint(150, 200)
        mil = 1
    return content, mil


def random_auged_bg():
    idx = np.random.randint(100)
    x = np.random.randint(abg[aname[idx]].shape[1] - 350)
    y = np.random.randint(abg[aname[idx]].shape[0] - 200)
    bg = abg[aname[idx]][y:y + 200, x:x + 350, :]
    bg = image.random_rotation(bg, 10)

    return bg


def aug_img(img):
    annotations = {'image': img}
    aug = albumentations.Compose([
        albumentations.GaussNoise(p=1),
        albumentations.MotionBlur(p=1),
        albumentations.Rotate(5),
        albumentations.OpticalDistortion(p=1),
        albumentations.IAAPerspective(scale=(0.005, 0.01), p=1),
        albumentations.CLAHE(p=1),
        albumentations.RandomBrightnessContrast(p=1)
    ])
    augmented = aug(**annotations)
    return augmented['image']


def paint_text(text, w, p, aug=False, test=False, useabg=False, randfont=False):
    img, white = random_bg()
    mil = (np.random.randn() > 0.8) & white & ord(text[-1]) > 300
    # a = cv2.putText(bg_im)
    if mil:
        r = g = b = 0
    else:
        c = np.random.randn()
        if c < 0.4:
            r = g = b = 0
        elif c < 0.75:
            r = 0
            g = 100
            b = 10
        else:
            r = g = 0
            b = 128
    cv2.rectangle(img, (0, 0), (266, 115), (r, g, b), 4)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    if not randfont:
        choice = 0
        f = font
        f2 = font2
    else:
        choice = np.random.randint(len(fonts))
        # choice = 2
        f = fonts[choice]
        f2 = fonts2[choice]

    if len(text) == 7:
        upper_off = upper_offsets[choice][7]
    elif len(text) == 6:
        upper_off = upper_offsets[choice][6]
    elif len(text) == 5:
        upper_off = upper_offsets[choice][5]
    elif len(text) == 4:
        upper_off = upper_offsets[choice][4]
    else:
        upper_off = upper_offsets[choice][-1]

    draw.text((upper_off, upper_offsets[choice][0]), text, font=f, fill=(r, g, b, 255))
    province = provinces[p]
    draw.text((offsets[p] + np.random.randint(-5, 5), upper_offsets[choice]['p']), province, font=f2, fill=(r, g, b, 255))

    if useabg:
        img_pil = img_pil.resize((np.random.randint(120, 290), np.random.randint(90, 150)))
        degree = np.random.randint(-20, 20)
        mask = Image.new('L', img_pil.size, 255)
        img_pil = img_pil.rotate(degree, expand=True)
        mask = mask.rotate(degree, expand=True)
        bg = Image.fromarray(random_auged_bg())
        bg.paste(img_pil, (np.random.randint(50), np.random.randint(50)), mask=mask)
        img_pil = bg.resize((270, 120))

    img = np.array(img_pil)
    if np.random.randn() > 0.75 and not useabg:
        img = image.random_rotation(img, 3 * w / w + 1)
    if aug:
        img = aug_img(img)
    alpha = 255
    if not test:
        img = img.astype('float32') / 255.
        alpha = 1.

    if mil:
        img = alpha - img
    return img
