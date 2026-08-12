import sys
from PIL import Image

args = sys.argv

img = Image.open(args[1])
# print(img.text)
for ele in img.text.items():
    print(f"{ele[0]}:{ele[1]}")
