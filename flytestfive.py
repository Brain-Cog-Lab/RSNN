from multi_robomaster import multi_robot
import time
import socket
import matplotlib;
matplotlib.use("TkAgg")
import numpy as np
import threading
import  random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def takeoff_all(robot_group):
    robot_group.mission_pad_on()
    robot_group.takeoff().wait_for_completed()
    robot_group.go({0: [0, 0, 110, 50, "m12"],1: [70, 70, 80, 50, "m12"],2: [-70, 70,100, 50, "m12"],3: [-70, -70,70, 50, "m12"],4: [70, -70,90, 50, "m12"]}).wait_for_completed()

def thr1():
    while(1):
        data, client_addr = server.recvfrom(BUFSIZE)
        #print(client_addr, data)
        data = str(data, encoding="utf8")
        xp = data.find('x')
        yp = data.find('y')
        zp = data.find('z')
        x = data[xp + 2:yp - 1]
        y = data[yp + 2:zp - 1]
        addr = client_addr[0]
        a = addr[len(addr) - 1]
        a = int(a)
        boids['pos'][a, 0] = int(x)
        boids['pos'][a, 1] = int(y)

def thr2():
    while(1):
        multi_drone.run([multi_drone_group1, update_boids])

def update_boids(robot_group):
    xs=boids['pos'][:, 0]
    ys=boids['pos'][:, 1]
    xvs=boids['vel'][:, 0]
    yvs=boids['vel'][:, 1]
    print('get position:',xs)
    print('get position:', ys)
    # Matrix off position difference and distance
    xdiff = np.add.outer(xs, -xs)
    ydiff = np.add.outer(ys, -ys)
    distance = np.sqrt(xdiff ** 2 + ydiff ** 2)

    # Calculate the boids that are visible to every other boid
    angle_towards = np.arctan2(-ydiff, -xdiff)
    angle_vel = np.arctan2(yvs, xvs)
    angle_diff = angle_towards - angle_vel[:, np.newaxis]
    visible = np.logical_and(angle_diff < np.pi / 4, angle_diff > -np.pi / 4) #visible based on the velocity direction
    # #update the velocity direction
    collision = np.clip(2.0 - distance / COLLISION_THRE, 1.0, 2.0) * visible #in the visible sight, the distance smaller than threshold will be marked as collision
    c_tmp=np.diag(np.diag(collision))
    collision=collision-c_tmp
    collision[collision==1]=0#onlybstacle the non-self o will be marked as collision
    #number of obstacles need to be avoided urgently
    xdiff_tmp=xdiff.copy()
    xdiff_tmp[np.where(collision==0)]=WORLD_WIDTH*WORLD_WIDTH
    index_min=(np.argmin(abs(xdiff_tmp) , axis=1))
    xdiff_use=np.array([xdiff_tmp[i,:][index_min[i]] for i in range(len(index_min))]) #if agent has multiple obstacles,only get the most nearest one
    xdiff_use[np.where(xdiff_use==WORLD_WIDTH*WORLD_WIDTH)]=0
    #print(collision,xdiff_use)
    xvs[np.where(xdiff_use<0)]=-1*SPEED#if agent is to the left of the obstacle, it moves toward left, If it's in the other direction, do the same thing
    xvs[np.where(xdiff_use>0)] = SPEED
    ydiff_tmp = ydiff.copy()
    ydiff_tmp[np.where(collision == 0)] = WORLD_WIDTH * WORLD_WIDTH
    index_miny = np.argmin(abs(ydiff_tmp), axis=1)
    ydiff_use=np.array([ydiff_tmp[i,:][index_miny[i]] for i in range(len(index_miny))])
    ydiff_use[np.where(ydiff_use==WORLD_WIDTH*WORLD_WIDTH)]=0
    yvs[np.where(ydiff_use<0)] = -1*SPEED
    yvs[np.where(ydiff_use>0)] = SPEED
    #
    # Wall avoidance  just give the reverse velocity
    xvs[np.where(xs<WALL_COLLISION_LIMIT-WORLD_WIDTH)] = SPEED
    yvs[np.where(ys < WALL_COLLISION_LIMIT-WORLD_WIDTH)] =SPEED
    xvs[np.where((WORLD_WIDTH - xs)<WALL_COLLISION_LIMIT)] = -1*SPEED
    yvs[np.where((WORLD_WIDTH - ys) < WALL_COLLISION_LIMIT)] = -1*SPEED
    #
    # #update position
    xs+=xvs
    ys+=yvs
    print('velocity x:', xvs)
    print('velocity y:', yvs)
    robot_group.go({0: [xs[0], ys[0], 110, 10, "m12"], 1: [xs[1], ys[1], 80, 10, "m12"],2: [xs[2], ys[2], 100,10, "m12"],3: [xs[3], ys[3], 70, 10, "m12"],4: [xs[4], ys[4], 90, 10, "m12"]}).wait_for_completed()
    if len(np.where(xs>200)[0])>0 or len(np.where(xs<-200)[0])>0 or len(np.where(ys>200)[0])>0 or len(np.where(ys<-200)[0])>0:
        robot_group.land().wait_for_completed()

N =5
WORLD_WIDTH = 140
COLLISION_THRE = 80
WALL_COLLISION_LIMIT=30
BUFSIZE = 1024
SPEED=20
ip_port = ('0.0.0.0', 8890)

#initialize position and velocity
global boids
boids = np.zeros(N, dtype=[('pos', int, 2), ('vel', int, 2)])
boids['pos']=np.array([[0,0],[70,70],[-70,70],[-70,-70],[70,-70]])

boids['vel'] = np.random.randint(-1, 2, (N, 2))
for i_vel in range(len(boids['vel'])):
    while(boids['vel'][i_vel][0]==0 and boids['vel'][i_vel][1]==0):
        boids['vel'][i_vel] = np.random.randint(-1, 2, (1, 2))
print('velocity', boids['vel'])
boids['vel']=boids['vel']*SPEED
#initialize multi-drones
#robot_sn_list = ['0TQZHBSCNT0DJX','0TQZHBSCNT0DCK','0TQZHBSCNT0DCL','0TQZHBSCNT0DC2','0TQZHBSCNT0DEN','0TQZHBSCNT0DRD']
robot_sn_list = ['0TQZHBSCNT0DJX','0TQZHBSCNT0DCK','0TQZHBSCNT0DCL','0TQZHBSCNT0DC2','0TQZHBSCNT0DEN']
multi_drone = multi_robot.MultiDrone()
multi_drone.initialize(robot_num=5)
multi_drone.number_id_by_sn([0, robot_sn_list[0]],[1, robot_sn_list[1]],[2, robot_sn_list[2]],[3, robot_sn_list[3]],[4, robot_sn_list[4]])
multi_drone_group1 = multi_drone.build_group([0,1,2,3,4])

#take off first, then open udp
multi_drone.run([multi_drone_group1, takeoff_all])
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # udp协议
server.bind(ip_port)

#multi threading
t1 = threading.Thread(target=thr1, args=())
t2 = threading.Thread(target=thr2, args=())
t1.start()
t2.start()