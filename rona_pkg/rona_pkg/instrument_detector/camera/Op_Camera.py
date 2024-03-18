import cv2
import os
import datetime
from instrument_detector.object_detector.Rona_mid.utils.general import imwrite
from instrument_detector.object_detector.general_utils.img_util import crop_image, crop_image_from_center, scale_image
import threading, queue

'''
class TakeCameraLatestPictureThread(threading.Thread):
    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        super().__init__()
        # Start thread
        self.start()
    def run(self):
        while True:
            ret, self.frame = self.camera.read()
'''

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name, cam_frame_w, cam_frame_h):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(3, cam_frame_w)  # width 1920
        self.cap.set(4, cam_frame_h)  # height 1080
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) #buffer size
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()


class OP_Cam:
    def __init__(self, cam_config):
        self.cam_port = cam_config["cam_port"]
        self.cam_frame_w = cam_config["cam_frame_width"]
        self.cam_frame_h = cam_config["cam_frame_height"]
        self.cam_save_frame = cam_config["cam_save_frame"]
        self.cam_frame_path = cam_config["cam_frame_path"]
        self.cam_frame_output_sz = cam_config["cam_frame_output_sz"]



    def camera_start(self):
        if self.cam_save_frame:
            if not os.path.exists(self.cam_frame_path):
                os.makedirs(self.cam_frame_path)
        #cam = cv2.VideoCapture(self.cam_port)
        #cam.set(3, self.cam_frame_w)  # width 1920
        #cam.set(4, self.cam_frame_h)  # height 1080
        #cam.set(cv2.CAP_PROP_BUFFERSIZE, 1) #buffer size
        cam = VideoCapture(self.cam_port , self.cam_frame_w, self.cam_frame_h)
        #self.thr_obj = TakeCameraLatestPictureThread(cam)
        return cam

    def camera_read_frame(self, cam):
        #ret, frame = cam.read()
        frame = cam.read()
        #frame = imread("2023-01-30-120023.jpg")
        # rescale and crop from center
        cam_frame = frame.copy()
        frame = scale_image(frame,scale_percent=100)
        frame = crop_image_from_center(frame, self.cam_frame_output_sz)

        if self.cam_save_frame:
            e = datetime.datetime.now()
            frame_name = "frame_" + e.strftime("%Y%m%d_%H%M%S") + "_camera.png"
            output_path = os.path.join(self.cam_frame_path, frame_name)
            imwrite(output_path, cam_frame)
            frame_name = "frame_" + e.strftime("%Y%m%d_%H%M%S") + "_mid_input.png"
            output_path = os.path.join(self.cam_frame_path, frame_name)
            imwrite(output_path, frame)

        return frame


    def camera_stop(self, cam):
        cam.release()