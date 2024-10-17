import rospy
import numpy as np
import cv2
from collections import deque, defaultdict
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Pose, Point
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from cv_bridge import CvBridge

map_received = False
odom_received = True

bot_map_coordinates = [0,0]

map_msg = None
current_map = None

marker_pub = None

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


def explore_cluster_bfs(points, start_point, max_neighbor_distance):
    n = len(points)
    tree = cKDTree(points)
    
    visited = set()
    queue = deque([(start_point, 0)])  # (point, distance)
    path = [start_point]
    path_distances = [0]
    
    farthest_point = start_point
    max_path_distance = 0
    
    while queue:
        current_point, current_distance = queue.popleft()
        
        # Find neighbors within max_neighbor_distance
        indices = tree.query_ball_point(current_point, max_neighbor_distance)
        
        for idx in indices:
            if idx not in visited:
                visited.add(idx)
                neighbor = points[idx]
                distance_to_neighbor = np.linalg.norm(neighbor - current_point)
                total_distance = current_distance + distance_to_neighbor
                
                queue.append((neighbor, total_distance))
                path.append(neighbor)
                path_distances.append(total_distance)
                
                if total_distance > max_path_distance:
                    max_path_distance = total_distance
                    farthest_point = neighbor
    
    return np.array(path), path_distances, farthest_point, max_path_distance


def pointToGrid(x, y, map_msg):
    map_x = int(round((x - map_msg.info.origin.position.x) / map_msg.info.resolution))
    map_y = int(round((y - map_msg.info.origin.position.y) / map_msg.info.resolution))
    return [map_x, map_y]

def gridToPoint(x, y, map_msg):
    world_x = x*map_msg.info.resolution + map_msg.info.origin.position.x
    world_y = y*map_msg.info.resolution + map_msg.info.origin.position.y
    return [world_x, world_y]

def odomCallback(msg):
    global bot_map_coordinates, odom_received, map_received, map_msg

    if map_received == True:
        bot_world_coordinates = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        bot_map_coordinates = pointToGrid(bot_world_coordinates[0], bot_world_coordinates[1], map_msg)

        odom_received = True

def mapCallback(msg):
    global map_msg, current_map, map_received

    rows = msg.info.width
    cols = msg.info.height

    map_data = np.zeros((rows, cols), dtype=int)

    for i in range(cols):
        for j in range(rows):
            index = j + i * rows
            if msg.data[index] == 100:
                map_data[j, i] = 1
            elif msg.data[index] == -1:
                map_data[j, i] = 0
            else:
                map_data[j, i] = 0

    map_msg = msg
    current_map = map_data
    map_received = True

def find_closest_one(grid, start_x, start_y):
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1,1), (-1, -1), (-1, 1), (1,-1)]
    
    # Initialize queue with starting point and set of visited nodes
    queue = deque([(start_x, start_y)])
    visited = set()
    visited.add((start_x, start_y))
    
    # BFS loop
    while queue:
        x, y = queue.popleft()
        
        # Check if current cell is the target
        if grid[x][y] == 1:
            return (x, y)
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check boundaries and if the cell has been visited
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    # Return None if no 1 is found
    return None

def publishGoalMarker(x, y, id):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "local_goal"
    marker.id = id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.5  
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.color.r = 1.0  
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    return marker

def main():
    global marker_pub, bot_map_coordinates, current_map, id

    rospy.init_node('local_goal_planner_py')

    # marker_pub = rospy.Publisher('bounding_box', Marker, queue_size=10)
    marker_pub = rospy.Publisher('local_goal', Marker, queue_size=10)

    rospy.Subscriber('/local_map', OccupancyGrid, mapCallback)
    rospy.Subscriber('/odom', Odometry, odomCallback)

    pub = rospy.Publisher('lanes', Float32MultiArray, queue_size=10)

    bridge = CvBridge()

    rate = rospy.Rate(10) 
    while not rospy.is_shutdown():

        if(map_received and odom_received):
            current_map = current_map.astype(np.uint8)

            skeleton = skeletonize(current_map)

            lanes = np.argwhere(skeleton == 1)

            msg = Float32MultiArray()
        
            # Set up the layout
            msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            msg.layout.dim[0].label = "height"
            msg.layout.dim[0].size = lanes.shape[0]
            msg.layout.dim[0].stride = lanes.shape[0] * lanes.shape[1]
            msg.layout.dim[1].label = "width"
            msg.layout.dim[1].size = lanes.shape[1]
            msg.layout.dim[1].stride = lanes.shape[1]
            
            # Copy the data
            msg.data = lanes.flatten().tolist()

            # Publish the message
            pub.publish(msg)



            # closest_lane_x, closest_lane_y = find_closest_one(skeleton, bot_map_coordinates[0], bot_map_coordinates[1])

            # # print(lanes.shape)

            # max_distance = 5

            # exploration_path, path_distances, farthest_point, max_path_distance  = explore_cluster_bfs(lanes, [closest_lane_x, closest_lane_y], max_distance)
            # # farthest_point = exploration_path[exploration_path.shape[0] - 1]
            
            # print(farthest_point)


            # point = gridToPoint(farthest_point[0], farthest_point[1], map_msg)

            # marker = publishGoalMarker(point[0], point[1], 1)
            # marker_pub.publish(marker)


        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
