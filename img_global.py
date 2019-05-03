HEIGHT, WIDTH, CHANNELS = 120, 160, 3
IMG_SHAPE = (HEIGHT, WIDTH, CHANNELS)
OUTPUT_SHAPE=3

import numpy as np
from PIL import Image

image=Image.open('./1.jpg')
array=np.array(image)
print(array)