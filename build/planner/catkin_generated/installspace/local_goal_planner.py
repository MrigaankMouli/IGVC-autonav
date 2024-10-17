import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Pose, Point
from visualization_msgs.msg import Marker
import cv2 as cv
import os

# Global Variables
local_map_msg = None
current_local_map = None
cols = 0
rows = 0

local_map_received = False
odom_received = False
local_goal_published = False

bot_world_coordinates = [0, 0]
bot_map_coordinates = [0, 0]

id = 112

marker_pub = None

def worldCoordinatesToLocalMapIndex(x, y):
    map_x = int(round((x - local_map_msg.info.origin.position.x) / local_map_msg.info.resolution))
    map_y = int(round((y - local_map_msg.info.origin.position.y) / local_map_msg.info.resolution))
    return [map_x, map_y]

def mapIndexToWorldCoordinates(x, y):
    world_x = x*local_map_msg.info.resolution + local_map_msg.info.origin.position.x
    world_y = y*local_map_msg.info.resolution + local_map_msg.info.origin.position.y
    return [world_x, world_y]

def checkValidity(map_x, map_y):
    return 0 <= map_x < rows and 0 <= map_y < cols

def mapCallback(msg):
    global local_map_msg, current_local_map, rows, cols, local_map_received

    rows = msg.info.width
    cols = msg.info.height

    map_data = np.zeros((rows, cols), dtype=int)

    for i in range(cols):
        for j in range(rows):
            index = j + i * rows
            if msg.data[index] == 100:
                map_data[j, i] = 1
            elif msg.data[index] == -1:
                map_data[j, i] = 2
            else:
                map_data[j, i] = 0

    local_map_msg = msg
    current_local_map = map_data
    local_map_received = True

    # rospy.loginfo(f"Local Map Received: {msg.info.width} x {msg.info.height}")

def odomCallback(msg):
    global bot_world_coordinates, bot_map_coordinates, odom_received

    bot_world_coordinates = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    bot_map_coordinates = worldCoordinatesToLocalMapIndex(bot_world_coordinates[0], bot_world_coordinates[1])

    if checkValidity(bot_map_coordinates[0], bot_map_coordinates[1]):
        # rospy.loginfo(f"Robot coordinates: ({bot_world_coordinates[0]}, {bot_world_coordinates[1]}) -> Map indices: ({bot_map_coordinates[0]}, {bot_map_coordinates[1]})")
        odom_received = True
    # else:
    #     rospy.logwarn(f"Robot coordinates: ({bot_world_coordinates[0]}, {bot_world_coordinates[1]}) are out of map bounds!")

def createBoundingBoxMap(center_x, center_y, box_size):
    half_size = int(box_size/2)
    
    map = []
    for dx in range(-half_size, half_size):
        row = []
        for dy in range(-half_size, half_size):
            x = center_x + dx
            y = center_y + dy
            if checkValidity(x, y):
                row.append(current_local_map[x, y])
            else:
                row.append(-1)
        map.append(row)
    
    return map
    

def createBoundingBox(x, y, size):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "bounding_box"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1 
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Create square points
    half_size = size / 2
    points = [
        Point(x - half_size, y - half_size, 0),
        Point(x + half_size, y - half_size, 0),
        Point(x + half_size, y + half_size, 0),
        Point(x - half_size, y + half_size, 0),
        Point(x - half_size, y - half_size, 0) 
    ]
    marker.points = points

    return marker

def publishGoalMarker(x, y):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "local_goal"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.5  # Increased size for visibility
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.color.r = 1.0  # Changed color to red for visibility
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    return marker

def main():
    global marker_pub, bot_map_coordinates, current_local_map, id

    rospy.init_node('local_goal_planner_py')

    # marker_pub = rospy.Publisher('bounding_box', Marker, queue_size=10)
    marker_pub = rospy.Publisher('local_goal', Marker, queue_size=10)

    rospy.Subscriber('/local_map', OccupancyGrid, mapCallback)
    rospy.Subscriber('/odom', Odometry, odomCallback)

    # rospy.spin()

    rate = rospy.Rate(1.5)  # 1 Hz
    while not rospy.is_shutdown():

        if (odom_received and local_goal_published == False):
            bounding_box_array = np.array(createBoundingBoxMap(bot_map_coordinates[0], bot_map_coordinates[1], 256))
            print(bounding_box_array.shape)

            array_2d = bounding_box_array

            array_3d = np.zeros((array_2d.shape[0], array_2d.shape[1], 3), dtype=np.float32)
            array_3d[array_2d == 2, 0] = 1
            array_3d[array_2d == 0, 1] = 1  
            array_3d[array_2d == 1, 2] = 1

            id = id + 1

            save_dir = os.path.expanduser("~/bounding_boxes/test")
            
            # Create the directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            file_name = f"bounding_box_{id}.npy"
            file_path = os.path.join(save_dir, file_name)

            np.save(file_path, array_3d)

            rospy.loginfo(f"{file_name} Array Saved")


        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass