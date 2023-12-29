import threading
import rospy
import os
from std_msgs.msg import Int32


HOST_NAME = os.environ["VEHICLE_NAME"]

class BotState:
    """
    defines all states needed to execute the entire project
    """
    
    def __init__(self, goal_stall):
        """
        goal_stall (int) - between 1 and 4, specifies the goal stall for parking
        """
        self.goal_stall = goal_stall
        self.lock = threading.Lock()
        # handy flags to turn on/off some robot behaviors
        self.lane_follow = True  # if true, do lane following

    
    def get_flags(self):
        self.lock.acquire()
        flags = {
            'lane_follow': self.lane_follow,
        }
        self.lock.release()
        return flags

    def get_lane_following_flag(self):
        self.lock.acquire()
        flag = self.lane_follow
        self.lock.release()
        return flag
