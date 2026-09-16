"""Render the site's typography-based sharing card (Pillow required)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
image = Image.new('RGB', (1200, 630), '#faf9f5')
draw = ImageDraw.Draw(image)
accent, ink, muted = '#df8065', '#252522', '#77756f'
latin = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
chinese = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

draw.rounded_rectangle((38, 38, 1162, 592), radius=24, outline='#e6e2d8', width=2)
draw.rounded_rectangle((82, 92, 132, 99), radius=3, fill=accent)
draw.text((150, 79), 'PERSONAL NOTES', fill=muted, font=ImageFont.truetype(latin, 22))
draw.text((80, 205), 'LittleHamster Xu', fill=ink, font=ImageFont.truetype(bold, 64))
draw.text((84, 307), '记录技术，也记录折腾。', fill=muted, font=ImageFont.truetype(chinese, 34))
draw.line((84, 451, 1116, 451), fill='#e6e2d8', width=2)
draw.text((84, 490), '技术实践  /  学习记录  /  日常想法', fill=muted, font=ImageFont.truetype(chinese, 23))
draw.text((990, 483), '{ }', fill=accent, font=ImageFont.truetype(bold, 42))
image.save(ROOT / 'static/images/default-share.png', optimize=True)
