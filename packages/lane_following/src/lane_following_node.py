#!/usr/bin/env python3
import json
import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32, Int32, String
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from sensor_msgs.msg import Range
import os
import threading
import math
import deadreckoning
import state_machine
import torch
from ultralytics import YOLO
from rospkg import RosPack

# from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern

HOST_NAME = os.environ["VEHICLE_NAME"]
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = False
ENGLISH = False

torch.backends.cudnn.enabled = False
torch.cuda.is_available = lambda: False


class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = HOST_NAME
        self.jpeg = TurboJPEG()
        self.loginfo("Initialized")

        # load model
        self.ros_pack = RosPack()
        self.model = YOLO(self.ros_pack.get_path("lane_following") + "/data/best.pt")
        self.model.to('cpu')

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220
        self.velocity = 0.34
        self.speed = .6
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.046
        self.D = -0.004
        self.last_error = 0
        self.last_time = rospy.get_time()

        # handling stopping at stopline
        self.stop_cnt = 6
        self.yolo_cnt = 0
        self.lock = threading.Lock()  # used to coordinate the subscriber thread and the main thread
        self.controller = deadreckoning.DeadReckoning()  # will handle wheel commands during turning

        # Publishers & Subscribers
        self.bot_state = state_machine.BotState()  # pass in 1 as placeholder; in the end self.bot_state is not used in part 3
        self.class_dict = {0: "Hsiao", 1: "Roast duck", 2: "arduino", 3: "balltank", 4: "base", 5: "duck", 6: "tank",
                           7: "terrorist"}
        self.led_map = {0: "RED", 1: "GREEN", 2: "BLUE", 3: "YELLOW", 4: "POPO", 5: "CYAN",
                        6: "WHITE",
                        7: "PURPLE"}
        if DEBUG:
            self.pub = rospy.Publisher("f/{self.veh}/output/image/mask/compressed",
                                       CompressedImage,
                                       queue_size=1)
        self.pub_led = rospy.Publisher(f"/{self.veh}/led_emitter_node/change_led", String, queue_size=1)

        self.sub_iof = rospy.Subscriber(f"/{self.veh}/front_center_tof_driver_node/range", Range, self.cb_iof,
                                        queue_size=1)

        self.sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage,
                                    self.callback, queue_size=1, buff_size="20MB")

        self.vel_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

    def cb_iof(self, msg):
        # print(msg.min_range, msg.max_range, msg.range)
        if self.bot_state.gey_obstacle_flag():
            return

        if msg.range <= 0.17:
            self.stop_cnt -= 1
        else:
            self.stop_cnt = 6

        if self.stop_cnt <= 0:
            if self.bot_state.get_lane_following_flag():
                self.bot_state.update_state("obstacle")
                # print("range:", msg.range)
            self.stop_cnt = 6

    def set_LED(self, cid):

        if 0 <= cid <= 7:
            print("change led to " + self.led_map[cid])
            self.pub_led.publish(self.led_map[cid])
        else:
            print("change led to OBSTACLE_STOPPED")
            self.pub_led.publish("OBSTACLE_STOPPED")

    def avoidance(self):
        # self.controller.driveForTime(-1 * self.speed, 1 * self.speed, 23)
        # self.controller.stop(10)
        # self.controller.driveForTime(1 * self.speed, 1 * self.speed, 20)
        # self.controller.stop(10)
        # self.controller.driveForTime(1 * self.speed, -1 * self.speed, 20)
        # self.controller.stop(10)
        # self.controller.driveForTime(1 * self.speed, 1 * self.speed, 50)
        # self.controller.stop(10)
        # self.controller.driveForTime(1 * self.speed, -1 * self.speed, 20)
        # self.controller.stop(10)
        # self.controller.driveForTime(1 * self.speed, 1 * self.speed, 18)

         self.controller.driveForTime(0.04 * self.speed, 0.9 * self.speed, 40)
         self.controller.driveForTime(0.5 * self.speed, 0.5 * self.speed, 10)
         self.controller.driveForTime(0.8 * self.speed, 0.1 * self.speed, 25)
         self.controller.driveForTime(0.6 * self.speed, 0.6 * self.speed, 25)
         self.controller.driveForTime(0.9 * self.speed, 0.1 * self.speed, 20)
         self.controller.driveForTime(0.3 * self.speed, 0.3 * self.speed, 20)
         self.controller.stop(5)

    def forward(self):
        self.controller.driveForTime(1 * self.speed, 1 * self.speed, 60)
        self.controller.stop(10)

    def callback(self, msg):

        img = self.jpeg.decode(msg.data)  # 480 680 3
        # yolo_crop = img[0:360, 170:510, :]

        if self.bot_state.gey_obstacle_flag():
            self.controller.reset_position()
            self.controller.stop(5)

            self.yolo_cnt -= 1
            if self.yolo_cnt > 0:
                return
            self.yolo_cnt = 10

            self.controller.driveForTime(-0.9 * self.speed, -0.9 * self.speed, 15)
            self.controller.stop(10)

            results = self.model.predict(img, show=False, device="cpu", conf=0.5)
            
            if len(results[0].boxes) == 0:
                self.set_LED(-1)
                print("no object detected")
                self.avoidance()
                self.bot_state.update_state("lane_follow")
                return
            max_conf = -1
            detected_cid = -1
            for boxes in results[0].boxes:
                cid = boxes.cls.tolist()[0]
                conf = boxes.conf.tolist()[0]

                if conf > max_conf:
                    max_conf = conf
                    detected_cid = cid

            print("max conf:", max_conf)
            print("find:" + self.class_dict[detected_cid])
            self.set_LED(detected_cid)
            if detected_cid >= 4:
                self.forward()
            else:
                self.avoidance()

            self.bot_state.update_state("lane_follow")
            return

        if not self.bot_state.get_lane_following_flag():
            self.proportional = None
            return

        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00']) + 15
                cy = int(M['m01'] / M['m00'])
                threshold = 200

                self.proportional = min(threshold, max(-threshold, cx - int(crop_width / 2) + self.offset))

                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = -100  # assume off to the right

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
        else:

            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            self.twist.v = self.velocity
            self.twist.omega = P + D
            if DEBUG:
                self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

        if self.bot_state.get_lane_following_flag():
            self.vel_pub.publish(self.twist)
    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)

    def on_shutdown(self):
        self.hook()


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()
