import threading
import rospy
import os
from std_msgs.msg import Int32

HOST_NAME = os.environ["VEHICLE_NAME"]


class BotState:
    """
    defines all states needed to execute the entire project
    """

    def __init__(self):
        """
        goal_stall (int) - between 1 and 4, specifies the goal stall for parking
        """
        # handy flags to turn on/off some robot behaviors
        self.lock = threading.Lock()
        self.flags = {
            'lane_follow': True,
            'obstacle': False,
        }

    def update_state(self, state_name):
        self.lock.acquire()
        if state_name not in self.flags:
            print("illegal state name")
            return
        for key in self.flags:
            self.flags[key] = False
        self.flags[state_name] = True
        self.lock.release()

    def get_flags(self):
        self.lock.acquire()
        flags = self.flags
        self.lock.release()
        return flags

    def get_lane_following_flag(self):
        self.lock.acquire()
        flag = self.flags['lane_follow']
        self.lock.release()
        return flag

    def gey_obstacle_flag(self):
        self.lock.acquire()
        flag = self.flags['obstacle']
        self.lock.release()
        return flag
