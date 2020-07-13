
from hyperlpr_py3 import pipline as pp
import cv2


video = 'data/test.mp4'
save_path = 'output-video/MySaveVideo.avi'
pp.SimpleRecognizePlate_video(video, save_path)

print('-------------')



