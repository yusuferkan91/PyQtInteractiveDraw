from argparse import ArgumentParser
from collections import namedtuple
import json
import math
import numpy as np
import paho.mqtt.client as mqtt
import signal
import sys
import cv2
import os
import time
import datetime
import math

import scipy.spatial

class Detection_start(object):
    def __init__(self, *args, **kwargs):
        # OpenCV-related variables
        self.delay = 5
        self.frame = None
        self.detection_frame = None
        # Assembly part and defect areas
        self.frame_ok_count = 0
        self.frame_defect_count = 0
        self.max_width = 0
        self.min_width = 0
        self.min_length = 0
        self.max_length = 0
        self.nextimage = list()
        self.prev_seen = False
        self.prev_defect = False
        self.total_parts = 0
        self.total_defect = 0
        self.one_pixel_length = 0
        self.diagonal_length_of_image_plane = 0
        self.kernel_count1 = 5
        self.kernel_count2 = 21
        self.threshold_count = 150
        # Define mqtt variables
        self.topic = "defects/counter"
        self.host = "localhost"
        self.port = 1883
        self.alive = 45

        # AssemblyInfo contains information about assembly line defects
        self.assembly_info = namedtuple("AssemblyInfo", "inc_total, defect, area, length, width, show, rects")
        self.info2 = self.assembly_info(inc_total="false", defect="false", area="0", length="0", width="0", show="false", rects=[])

    def update_info(self, info1):

        self.info2 = self.AssemblyInfo(inc_total=info1.inc_total, defect=info1.defect, area=info1.area, length=info1.length,
                             width=info1.width, show=info1.show, rects=info1.rects)
        if info1.inc_total:
            self.total_parts += 1
        if info1.defect:
            self.total_defect += 1


    # Returns the most-recent AssemblyInfo for the application
    def getcurrent_info(self):
        current = self.info2
        return current


    # Returns the next image from the list
    def nextimage_available(self):
        rtn = None
        if self.nextimage:
            rtn = self.nextimage.pop(0)
        return rtn


    # Adds an image to the list
    def add_image(self, img):

        if not self.nextimage:
            self.nextimage.append(img)


    # Publish MQTT message with a JSON payload
    def messageRunner(self):
        info3 = self.getcurrent_info()
        # client.publish(topic, payload=json.dumps({"defect": info3.defect}))


    # Signal handler
    def signal_handler(self, sig, frame):
        cv2.destroyAllWindows()
        # client.disconnect()
        sys.exit(0)


    '''******************************************************************'''


    def linex(self, p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C


    def intersection(self, L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return False


    '''******************************************************************'''


    def euclidean_dist(self, po):
        tmp = np.sqrt((po[0][0] - po[1][0]) * (po[0][0] - po[1][0]) + (po[0][1] - po[1][1]) * (po[0][1] - po[1][1]))
        # tmp = tmp * 0.165/2.0 #(1920*1080, 1px = 0.165mm)
        tmp = tmp * 61.16 / 2678.0
        return round(tmp, 1)


    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy


    def get_length2(self, color_img, mask, right_pt, left_pt, ph, color):
        mask2 = mask * 0
        cv2.line(mask2, (mask2.shape[1] - 1, right_pt), (0, left_pt), 255, 1)
        mask = np.logical_and(mask, mask2)
        po2 = np.transpose(np.nonzero(mask))
        po = po2

        px1 = po2[0][1]
        px2 = po2[1][1]
        py1 = po2[0][0]
        py2 = po2[1][0]

        mask = mask.astype(np.uint8)
        mask = mask * 255
        d = self.euclidean_dist(po)

        xc = (po[0][1] + po[1][1]) // 2
        yc = (po[0][0] + po[1][0]) // 2

        if d > 3.5:
            cv2.line(color_img, (px1, py1), (px2, py2), color, 5)
            cv2.putText(color_img, str(d) + " mm", (xc + ph[0], yc + ph[1]), cv2.FONT_HERSHEY_DUPLEX, 2, color, 3)

        return color_img, xc, yc, px1, py1


    def get_length(self, mask, color_img, vx, vy, x, y, ph, color):
        left_pt = int((-x * vy / vx) + y)
        right_pt = int(((color_img.shape[1] - x) * vy / vx) + y)
        xc = 0
        yc = 0
        try:
            color_img, xc, yc, px1, py1 = self.get_length2(color_img, mask, right_pt, left_pt, ph, color)
        except Exception as e:
            print(e)

        return color_img, xc, yc


    def rotated_rect(self, src_gray, color_img):

        kernel = np.ones((self.kernel_count1, self.kernel_count1), np.uint8)
        kernel2 = np.ones((self.kernel_count2, self.kernel_count2), np.uint8)

        img = cv2.GaussianBlur(src_gray, (11, 11), 0)

        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        ret, canny_output = cv2.threshold(img, self.threshold_count, 255, cv2.THRESH_BINARY)
        # cv2.imshow("asd", canny_output)
        # cv2.waitKey(1)
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros(img.shape, np.uint8)
        try:

            for i, c in enumerate(contours):
                minRect = cv2.minAreaRect(c)
                box = cv2.boxPoints(minRect)
                cnt = contours[i]

                cv2.drawContours(mask, cnt, -1, 255, 1)
                vx, vy, x, y = cv2.fitLine(cnt, 1, 0, 0.01, 0.01)
                color_img, xc, yc = self.get_length(mask, color_img, vx, vy, x, y, [50, 100], (0, 255, 0))

                color_img, xc, yc = self.get_length(mask, color_img, -vy, vx, xc, yc, [-200, -200], (0, 0, 255))

        except Exception as e:
            print(f'error:{e}')

        return color_img, canny_output


    '''******************************************************************'''
    def capRead(self):
        ret, frame = self.capture.read()
        if not ret:
            return None
        # frame = cv2.resize(frame, (960, 540))
        frame, _ = self.rotated_rect(255 - cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), frame)

        # fig.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.pause(0.01)

        displayFrame = frame.copy()
        self.add_image(frame)

        # frameRunner()
        assemble_line = self.getcurrent_info()
        length = str("Expected length (mm): = [{} - {}]".format(self.min_length, self.max_length))
        width = str("Expected width (mm): = [{} - {}]".format(self.min_width, self.max_width))
        Measurement = "Area (mm * mm) : {}".format(assemble_line.area)
        Length = "Length (mm) : {}".format(assemble_line.length)
        Width = "Width (mm)  : {}".format(assemble_line.width)
        total_part = "Total_parts : {}".format(self.total_parts)
        total_defects = "Total_defects : {}".format(self.total_defect)
        defects = "Defect : {}".format("False")

        if assemble_line.show is True:
            cv2.rectangle(displayFrame, (assemble_line.rects[0], assemble_line.rects[1]), (
                assemble_line.rects[0] + assemble_line.rects[2], assemble_line.rects[1] + assemble_line.rects[3]),
                          (0, 0, 255), 2)
            defects = "Defect : {}".format("True")
            cv2.putText(displayFrame, defects, (5, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        else:
            if assemble_line.rects:
                cv2.rectangle(displayFrame, (assemble_line.rects[0], assemble_line.rects[1]), (
                    assemble_line.rects[0] + assemble_line.rects[2], assemble_line.rects[1] + assemble_line.rects[3]),
                              (0, 255, 0), 2)
                defects = "Defect : {}".format("False")
                cv2.putText(displayFrame, defects, (5, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(displayFrame, Measurement, (5, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(displayFrame, length, (5, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(displayFrame, width, (5, 220), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(displayFrame, Length, (5, 120), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(displayFrame, Width, (5, 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(displayFrame, total_part, (5, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(displayFrame, total_defects, (5, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(displayFrame, defects, (5, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        # cv2.imshow("Object size detector", cv2.resize(displayFrame, None, fx=0.35, fy=0.35))

        # print(type(self.detection_frame))
        if cv2.waitKey(self.delay) > 0:
            return None
        return displayFrame
    def main(self):
        print("main")
        self.capture = cv2.VideoCapture(0)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / fps)
        print("1")
        one_pixel_length = 0.0264583333
        buyuk_civata = 331.6  # mm

        signal.signal(signal.SIGINT, self.signal_handler)
        print("2")
        if self.max_length < self.min_length:
            print("\nInvalid Arguments: Max length of the object is less then Min length!\n")
            sys.exit(0)

        if self.max_width < self.min_width:
            print("\nInvalid Arguments: Max width of the object is less then Min width!\n")
            sys.exit(0)
        print("3")
        # Read video input data
        # while capture.isOpened():


        self.capture.release()

        # Destroy all the windows
        cv2.destroyAllWindows()


# import matplotlib.pyplot as plt
#
# if __name__ == '__main__':
#     # Create a new instance
#     # client = mqtt.Client("object_size_detector")
#     # # Connect to broker
#     # client.connect(host, port, alive)
#     fig = plt.imshow(np.zeros((2160, 3840), dtype=np.uint8))
#     main()
#     # Disconnect MQTT messaging
#     # client.disconnect()
