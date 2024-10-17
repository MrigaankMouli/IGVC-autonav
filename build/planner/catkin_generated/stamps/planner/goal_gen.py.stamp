import numpy as np
from sklearn.cluster import DBSCAN

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Pose, Point
from visualization_msgs.msg import Marker

local_map_received = False
odom_received = False
local_goal_published = False

bot_world_coordinates = [0, 0]
bot_map_coordinates = [0, 0]

local_map_msg = None
current_local_map = None

cols = 0
rows = 0

def pointToGrid(x, y):
    map_x = int(round((x - local_map_msg.info.origin.position.x) / local_map_msg.info.resolution))
    map_y = int(round((y - local_map_msg.info.origin.position.y) / local_map_msg.info.resolution))
    return [map_x, map_y]

def gridToPoint(x, y):
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
            else:
                map_data[j, i] = msg.data[index]

    local_map_msg = msg
    current_local_map = map_data
    local_map_received = True

def odomCallback(msg):
    global bot_world_coordinates, bot_map_coordinates, odom_received

    bot_world_coordinates = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    bot_map_coordinates = pointToGrid(bot_world_coordinates[0], bot_world_coordinates[1])

    if checkValidity(bot_map_coordinates[0], bot_map_coordinates[1]):
        odom_received = True

def createBoundingBoxMap(center_x, center_y, box_size):
    if box_size % 2 == 0:
        box_size += 1 
    
    half_size = box_size // 2
    
    map = []
    for dx in range(-half_size, half_size + 1):
        row = []
        for dy in range(-half_size, half_size + 1):
            x = center_x + dx
            y = center_y + dy
            if checkValidity(x, y):
                row.append(current_local_map[x, y])
            else:
                row.append(-1)
        map.append(row)
    
    return map

def calculate_cumulative_distances_2d(points):
    # Calculate the distance between consecutive points in 2D
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    
    # Calculate cumulative distance
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
    return cumulative_distances

def split_trajectory_2d(points, num_segments=10):
    # Calculate cumulative distances along the trajectory
    cumulative_distances = calculate_cumulative_distances_2d(points)
    
    # Total length of the trajectory
    total_length = cumulative_distances[-1]
    
    # Determine the segment length
    segment_length = total_length / num_segments
    
    # Split the points based on cumulative distance
    segments = []
    start_idx = 0
    
    for i in range(1, num_segments + 1):
        end_idx = np.searchsorted(cumulative_distances, segment_length * i)
        segments.append(points[start_idx:end_idx])
        start_idx = end_idx
    
    return segments

def compute_midpoint(segment1, segment2):
    # Compute the midpoints of the start and end points of the two segments
    midpoint1 = (segment1[0] + segment1[-1]) / 2
    midpoint2 = (segment2[0] + segment2[-1]) / 2
    return (midpoint1 + midpoint2) / 2

def segment_distance(segment1, segment2):
    # Compute the average distance between all pairs of points (one from segment1, one from segment2)
    distances = np.linalg.norm(np.expand_dims(segment1, 1) - np.expand_dims(segment2, 0), axis=2)
    return np.min(distances)

def find_closest_segments(lane1_segments, lane2_segments):
    midpoints = []
    
    for segment1 in lane1_segments:
        min_distance = float('inf')
        closest_segment2 = None
        
        for segment2 in lane2_segments:
            distance = segment_distance(segment1, segment2)
            
            if distance < min_distance:
                min_distance = distance
                closest_segment2 = segment2
        
        if closest_segment2 is not None:
            midpoint = compute_midpoint(segment1, closest_segment2)
            midpoints.append(midpoint)
    
    return np.array(midpoints)

def apply_transformation(midpoint, bot_map_coordinates):
    # Define the threshold
    threshold = 121 / 2
    
    # Apply the transformation logic
    if midpoint[0] < threshold:
        goal_x = bot_map_coordinates[0] + (threshold - midpoint[0])
    else:
        goal_x = bot_map_coordinates[0] - (threshold - midpoint[0])
    
    if midpoint[1] < threshold:
        goal_y = bot_map_coordinates[1] + (threshold - midpoint[1])
    else:
        goal_y = bot_map_coordinates[1] - (threshold - midpoint[1])

    goal_world_coords = gridToPoint(goal_x, goal_y)
    
    return goal_world_coords

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
    marker.scale.x = 0.5  # Increased size for visibility
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.color.r = 1.0  # Changed color to red for visibility
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    return marker

def main():
    global marker_pub, bot_map_coordinates, current_local_map, model, odom_received, local_map_received, local_goal_published

    rospy.init_node('local_goal_planner_py')

    # marker_pub = rospy.Publisher('bounding_box', Marker, queue_size=10)
    marker_pub = rospy.Publisher('local_goal', Marker, queue_size=10)

    rospy.Subscriber('/local_map', OccupancyGrid, mapCallback)
    rospy.Subscriber('/odom', Odometry, odomCallback)

    # rospy.spin()

    rate = rospy.Rate(10)  # 1 Hz
    while not rospy.is_shutdown():

        if (odom_received and local_map_received and not local_goal_published):

            bounding_box = createBoundingBoxMap(bot_map_coordinates[0], bot_map_coordinates[1], 121)
            bounding_box_array = np.array(bounding_box)

            lanes = np.argwhere(bounding_box_array == 1)

            dbscan = DBSCAN(eps = 10, min_samples = 20)
            labels = dbscan.fit_predict(lanes)

            clusters = []
            for label in np.unique(labels):
                if label != -1:  # Ignore noise points labeled as -1
                    cluster_points = lanes[labels == label].tolist()
                    clusters.append(cluster_points)

            clusters.sort(key=len, reverse=True)
            largest_cluster = np.array(clusters[0])
            second_largest_cluster = np.array(clusters[1])

            split_points_lane_one = split_trajectory_2d(largest_cluster, num_segments=10)
            split_points_lane_two = split_trajectory_2d(second_largest_cluster, num_segments=10)

            midpoints = find_closest_segments(split_points_lane_one, split_points_lane_two)

            transformed_midpoints = [apply_transformation(midpoint, np.array(bot_map_coordinates)) for midpoint in midpoints]

            id = 0

            for midpoint in transformed_midpoints:
                id += 1
                marker_pub.publish(publishGoalMarker(midpoint[0], midpoint[1], id))


        rate.sleep()

if __name__ == '__main__':
    main()
