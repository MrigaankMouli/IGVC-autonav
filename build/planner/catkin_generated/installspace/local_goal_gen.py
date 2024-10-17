import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import os
import gc

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Pose, Point
from visualization_msgs.msg import Marker

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.conv1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = DoubleConv(256, 512)
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv5 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        
        # Final Conv
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        # Bridge
        b = self.bridge(p3)
        
        # Decoder
        u1 = self.up1(b)
        c3 = self.center_crop(c3, u1)  
        u1 = torch.cat([u1, c3], dim=1)
        c4 = self.conv4(u1)
        u2 = self.up2(c4)
        c2 = self.center_crop(c2, u2)  
        u2 = torch.cat([u2, c2], dim=1)
        c5 = self.conv5(u2)
        u3 = self.up3(c5)
        c1 = self.center_crop(c1, u3)
        u3 = torch.cat([u3, c1], dim=1)
        c6 = self.conv6(u3)
        
        # Final Conv
        out = self.final_conv(c6)
        
        # Ensure output size matches input size
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(out)
    
    def center_crop(self, layer, target_size):
        _, _, h, w = target_size.size()
        diff_y = (layer.size()[2] - h) // 2
        diff_x = (layer.size()[3] - w) // 2
        return layer[:, :, diff_y:diff_y + h, diff_x:diff_x + w]


model = UNet()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("/home/asmit/catkin_ws/src/planner/scripts/unet_model_augmented.pth"))
model.to(device)

model.eval()

local_map_received = False
odom_received = False
local_goal_published = False

bot_world_coordinates = [0, 0]
bot_map_coordinates = [0, 0]

local_map_msg = None
current_local_map = None

cols = 0
rows = 0

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
            else:
                map_data[j, i] = msg.data[index]

    local_map_msg = msg
    current_local_map = map_data
    local_map_received = True

def odomCallback(msg):
    global bot_world_coordinates, bot_map_coordinates, odom_received

    bot_world_coordinates = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    bot_map_coordinates = worldCoordinatesToLocalMapIndex(bot_world_coordinates[0], bot_world_coordinates[1])

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
    global marker_pub, bot_map_coordinates, current_local_map, model, odom_received, local_map_received, local_goal_published, device

    rospy.init_node('local_goal_planner_py')

    # marker_pub = rospy.Publisher('bounding_box', Marker, queue_size=10)
    marker_pub = rospy.Publisher('local_goal', Marker, queue_size=10)

    rospy.Subscriber('/local_map', OccupancyGrid, mapCallback)
    rospy.Subscriber('/odom', Odometry, odomCallback)

    # rospy.spin()

    rate = rospy.Rate(10)  # 1 Hz
    while not rospy.is_shutdown():

        if (odom_received and local_map_received and local_goal_published == False):
            gc.collect()       # Run garbage collection
            torch.cuda.empty_cache()

            bounding_box = createBoundingBoxMap(bot_map_coordinates[0], bot_map_coordinates[1], 241)
            bounding_box_array = np.reshape(np.array(bounding_box), (1, 241, 241, 1))

            array_2d = np.squeeze(bounding_box_array)
        
            # Create a 3-channel array to store the RGB values
            array_3d = np.zeros((array_2d.shape[0], array_2d.shape[1], 3), dtype=np.float32)
            
            # Convert the 2D array values to RGB
            array_3d[array_2d == -1, 0] = 1  # Occupied space (red)
            array_3d[array_2d == 0, 1] = 1  # Unoccupied space (green)
            array_3d[array_2d == 1, 2] = 1
            array_3d = array_3d.reshape((1, 241, 241, 3))

            array_3d = torch.tensor(array_3d, dtype=torch.float32).to(device)
            array_3d = array_3d.permute(0, 3, 1, 2)  

            with torch.no_grad():
                predictions = model(array_3d)

            predictions = predictions.reshape((241, 241))
            predictions = np.array(predictions.cpu())

            np.save("/home/asmit/map.npy", np.array(array_3d.cpu()))
            np.save("/home/asmit/goal.npy", predictions)
            print("Saved")

            # predictions = np.rot90(predictions)

            coordinates = np.argwhere(predictions > 0.7)
            coordinates = np.reshape(np.mean(coordinates, axis = 0), (2))

            # print(coordinates[0], coordinates[1])

            if(coordinates[0] < 241/2):
                goal_x = bot_map_coordinates[0] + (241/2 - coordinates[0])
            else :
                goal_x = bot_map_coordinates[0] - (241/2 - coordinates[0])

            if(coordinates[1] < 241/2):
                goal_y = bot_map_coordinates[1] + (241/2 - coordinates[1])
            else :
                goal_y = bot_map_coordinates[1] - (241/2 - coordinates[1])

            goal_world_coords = mapIndexToWorldCoordinates(goal_x, goal_y)

            print(goal_world_coords[0], goal_world_coords[1])

            marker = publishGoalMarker(goal_world_coords[0], goal_world_coords[1])

            marker_pub.publish(marker)

            # np.save("/home/asmit/map.npy", array_2d)
            # np.save("/home/asmit/goal.npy", predictions)

            # print(predictions)

            # print("Saved")

            odom_received = False
            local_map_received = False
            # local_goal_published = True



        rate.sleep()

if __name__ == '__main__':
    main()
