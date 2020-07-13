# coding: utf-8
# from tornado import web, httpserver, ioloop, websocket
import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import base64
# import face_recognition

# 定义一些基本设置
# port = 10101
# address = "localhost"
JPEG_HEADER = "data:image/jpeg;base64,"  # 这个是对图片转码用的

# 开启一个摄像头
cap = cv2.VideoCapture(0)


def get_image_dataurl(known_face_encodings, known_face_names, face_locations, face_encodings, face_names, process_this_frame):
    # 从摄像头读取数据, 读取成功 ret为True,否则为False,frame里面就是一个三维数组保存图像
    ret, frame = cap.read()

    # 将视频帧的大小调整为1/4以加快人脸识别处理
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
    rgb_small_frame = small_frame[:, :, ::-1]

    # 仅每隔一帧处理一次视频以节省时间
    if process_this_frame:
        # 查找当前视频帧中的所有面和面编码
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 查看该面是否与已知面匹配
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "未知"

            # 如果在已知的人脸编码中找到匹配项，请使用第一个。
            # 如果匹配为True：
            # first_match_index=matches.index(true)
            # name=known_face_names[first_match_index]

            # 或者，使用与新面的距离最小的已知面
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # 显示结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 由于我们在中检测到的帧被缩放到1/4大小，因此缩放备份面位置
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 在脸上画一个方框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 在面下画一个有名字的标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # cv2.putText 不支持中文所以这里换一种方法
        fontpath = "./font/simsun.ttc"  # 宋体字体文件
        font_1 = ImageFont.truetype(fontpath, 30)  # 加载字体, 字体大小
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((left + 10, bottom - 32), name, font=font_1, fill=(255, 255, 255))
        frame = np.array(img_pil)

    # 先将数组类型编码成 jepg 类型的数据,然后转字节数组,最后将其用base64编码
    r, buf = cv2.imencode(".jpeg", frame)
    dat = Image.fromarray(np.uint8(buf)).tobytes()
    img_date_url = JPEG_HEADER + str(base64.b64encode(dat))[2:-1]
    return img_date_url


class IndexHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        self.render("index.html")


class VideoHandler(websocket.WebSocketHandler):
    # 加载示例图片并学习如何识别它。
    obama_image = face_recognition.load_image_file("./img/jason.JPG")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # 加载第二个示例图片并学习如何识别它。
    biden_image = face_recognition.load_image_file("./img/davis.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # 创建已知面编码及其名称的数组
    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding
    ]
    known_face_names = [
        "张**",
        "赵**"
    ]

    # 初始化一些变量
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    # 处理接收数据
    def on_message(self, message):
        self.write_message({"img": get_image_dataurl(self.known_face_encodings, self.known_face_names, self.face_locations, self.face_encodings, self.face_names, self.process_this_frame)})


if __name__ == '__main__':
    app = web.Application(handlers=[(r"/", IndexHandler),
                                    (r"/index", IndexHandler),
                                    (r'/video', VideoHandler)],
                          template_path=os.path.join(os.path.dirname(__file__), "ui"))
    http_server = httpserver.HTTPServer(app)
    http_server.listen(port=port, address=address)
    print("URL: http://{}:{}/index".format(address, port))
    ioloop.IOLoop.instance().start()
