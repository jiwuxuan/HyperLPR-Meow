
from hyperlpr_py3 import pipline as pp
import cv2


image = 'data/009.jpg'
save_path = 'output-image/MySaveImage.jpg'
pp.SimpleRecognizePlate(image, save_path)

print('-------------')

