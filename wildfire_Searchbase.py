""" Search Based Planner - A* Algorithm Implementation"""

#Import essential libraries to be used 
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from queue import PriorityQueue
import matplotlib.pyplot as plt 
from numpy import pi, sqrt
from os import path
import numpy as np 
import threading
import random
import math
import time
import cv2

#Load Image for Ego Vehicle - Ackermann Steering Truck 
ego_veh_path = '/Users/anoushka/Desktop/Motion Planning /HOMEWORKS/Wildfire/trailer.png'

#Setting a scale_ratio for the Environment Display
sc_ratio = 15/3

#Setting Initial Truck positions 
ego_veh_start = [0,0,0,0,0]
ego_veh_goal = [-1,-1]

#Setting the Simulation Parameters to be recorded for visualization
sim_end = True
cnt_extinguish = 0
cpu_time = 0


class World :
    def __init__(self) -> None:
        # Initialise the Grid Environment parameters 
        self.width = round(250/sc_ratio)
        self.height =round(250/sc_ratio)
        # Grid Size for the Environment Display 
        self.grid_size = self.width*self.height
        # Obstacle Coverage percentage 
        self.coverage = 15
        # Size of Each Obstacle 
        self.obstacle_size = 4*round(15/sc_ratio)
        # Define the shapes of Tetriminoes obstacles to be places in the world
        self.obstacle = np.array([np.array([[1],[1],[1],[1]]),
                            np.array([[1,1,0],[0,1,1]]),
                            np.array([[1,1],[1,1]]),
                            np.array([[1,0],[1,0],[1,1]]),
                            np.array([[1,1,1],[0,1,0]]),
                            np.array([[0,1],[0,1],[1,1]]),
                            np.array([[0,1,1],[1,1,0]])],dtype=object)
        
        # Measuring State of each Obstacle 
        self.obstacle_state = []
        # Bruning Obstacle List -> Add the obstacles that are burning 
        self.burning_list = []
        # Initialising an Empty Grid 
        self.grid = np.zeros((self.width,self.height)) 

    # Checks if the given grid position is free of obstacle. 
    def is_occupied(self,grid,obs,width,height):       
        for i in range(width):
            for j in range(height):
                if grid[i][j] == 1 and obs[i][j] == 1:
                    #Return True if not empty
                    return True
        return False

    # Sorts the list of obstacle_state based on the x-coordinate of the obstacles. 
    # If two obstacles have the same x-coordinate, they are sorted based on their y-coordinate.
    def sort_obs(self):
        self.obstacle_state.sort(key=lambda obs: (obs[0], obs[1]))
        
    # Fills the grid with obstacles until the desired coverage percentage is reached.
    def create_grid(self, coverage):
        #The number of obstacles to be placed in the grid 
        num_obstacle = round(coverage / 100 * self.grid_size / self.obstacle_size)
        print("The coverage for the Environment: {f}".format(f=(num_obstacle*self.obstacle_size/self.grid_size)*100))
        
        #keeping track of obstacles placed
        tot_obstacles = 0
        while tot_obstacles < num_obstacle:
            #Choose a random Position to place the obstacle within the grid bounds
            i, j = random.randint(2, self.width - 2), random.randint(2, self.height - 2)
            #Choose random obstacle shape
            idx_obs = random.randint(0, len(self.obstacle) - 1)
            #Extract the obstacle shape
            obstacle = self.obstacle[idx_obs]
            o_w, o_h = obstacle.shape
            #Boundary and Occupancy Check 
            if i + o_w > self.width or j + o_h > self.height:
                continue
            elif self.is_occupied(self.grid[i:i+o_w, j:j+o_h], obstacle, o_w, o_h):
                continue
            m_d = 4 
            m = m_d + max(o_w, o_h) // 2
            if any(((i - obs_i)**2 + (j - obs_j)**2)**0.5 < m for obs_i, obs_j, _ in self.obstacle_state):
                continue

            #Place the Obstacle on the grid
            for obs_i in range(o_w):
                for obs_j in range(o_h):
                    if obstacle[obs_i][obs_j] == 1:
                        self.grid[i+obs_i][j+obs_j] = obstacle[obs_i][obs_j]
                        self.obstacle_state.append([i+obs_i, j+obs_j, "N"])
            #Keep a count of total obstacle placed
            tot_obstacles += 1
        #Sort the obstacles
        self.sort_obs()
        print("Created the Environment with Obstacles")
        return
    
class Robot_Kinematics():
    def __init__(self, world= World) -> None:
        self.world = world

        #Defining the kinematic parameters of the Robot Truck
        self.wheelbase = 3/sc_ratio
        self.steering_angle = 12
        self.vel = 10/sc_ratio
        self.truck_height = 2.2/sc_ratio
        self.truck_width = 4.9/sc_ratio
        
        #Burning and extinguishing fire Radius
        self.burn_rad = 30/sc_ratio
        self.extinguish_rad = 10 
        self.ego_veh_bound_T = []
        for i in range(3):
            row = []
            for j in range(4):
                if i == 0:
                    if j == 0:
                        bound = -(self.truck_width - self.wheelbase) / 2
                    else:
                        bound = self.truck_width - (self.truck_width - self.wheelbase) / 2
                elif i == 1:
                    if j == 0 or j == 1:
                        bound = -self.truck_height / 2
                    else:
                        bound = self.truck_height / 2
                else:
                    bound = 1
                row.append(bound)
            self.ego_veh_bound_T.append(row)


class Planner_Controller:
    def __init__(self,robot = Robot_Kinematics , world= World) -> None:
        self.robot = robot
        self.world = world

    # Utilizing the Kinematic Equations and Constraints to define the neighboring positions of the vehicle
    def get_neighbours(self, x, y, theta):
        # Empty list to store the valid neighbouring positions limited by the kinematic and non holonomic constraints
        neighbour = []
        for i in range(-self.robot.steering_angle, self.robot.steering_angle+1, 6):
            x_dot = self.robot.vel * math.cos(theta * (pi / 180))
            y_dot = self.robot.vel * math.sin(theta * (pi / 180))
            theta_dot = (self.robot.vel * math.tan(i * (pi / 180)) / self.robot.wheelbase) * (180 / pi)
            for dx, dy, dtheta, dv in [ (x_dot, y_dot, theta_dot, self.robot.vel),
                                        (-x_dot, -y_dot, -theta_dot, -self.robot.vel) ]:
                new_x = x + dx
                new_y = y + dy
                new_theta = (theta + dtheta) % 360
                if self.valid_point(new_x, new_y, new_theta):
                    neighbour.append([round(new_x, 2), round(new_y, 2), round(new_theta, 2), dv, i])
        return neighbour

    # Distance cost between two points
    def dist_cost(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        return distance

   
    #Estimated cost to reach goal position from current position
    def heuristic_function(self, x, y, turn, pre_turn, direction, pre_direction):
        turn_cost = int(turn != 0)
        change_turn_cost = int(turn != pre_turn) * 5
        reverse_cost = int(direction < 0) * 5
        gear_change = int(direction != pre_direction and pre_direction != 0) * 100
        distance_cost = 50 * sqrt((x - ego_veh_goal[0])**2 + (y - ego_veh_goal[1])**2) 
        heuristic = distance_cost + turn_cost + change_turn_cost + reverse_cost + gear_change
        return heuristic
    
   
    # Outline of Robot
    def get_boundary(self, x, y, theta):
        cos_theta = math.cos(theta * (math.pi / 180))
        sin_theta = math.sin(theta * (math.pi / 180))
        x_new = x
        y_new = y
        homogeneous_matrix = np.array([[cos_theta, -sin_theta, x_new], [sin_theta, cos_theta, y_new], [0, 0, 1]])
        mat_mul = np.dot(homogeneous_matrix, self.robot.ego_veh_bound_T)
        new_boundary = mat_mul[:2, :].T.tolist()
        return new_boundary

    # Check if the position is avaialble or not 
    def valid_point(self, x, y, theta):
        if x < self.robot.wheelbase or y < self.robot.wheelbase or x > self.world.width - self.robot.wheelbase - 2 or y > self.world.height - self.robot.wheelbase - 2:
            return False
        if self.world.grid[round(x)][round(y)] == 1:
            return False
        # Check for collision between boundary and obstacles
        boundary = self.get_boundary(x, y, theta)
        for bx, by in boundary:
            if not self.is_valid_boundary_point(round(bx), round(by)):
                return False

        return True

    def is_valid_boundary_point(self, x, y):
        if self.world.grid[x][y] == 1:
            return False
        if self.world.grid[x - 1][y] == 1:
            return False
        if self.world.grid[x][y - 1] == 1:
            return False
        if self.world.grid[x + 1][y] == 1:
            return False
        if self.world.grid[x][y + 1] == 1:
            return False
        if self.world.grid[x - 1][y - 1] == 1:
            return False
        if self.world.grid[x + 1][y - 1] == 1:
            return False
        if self.world.grid[x + 1][y + 1] == 1:
            return False
        if self.world.grid[x - 1][y + 1] == 1:
            return False

        return True

    # To select path with shortest disctance 
    def priority(self,queue): 
        min = math.inf
        index = 0
        for check in range(len(queue)):
            _,value,_,_ = queue[check]
            if value<=min:
                min = value
                index = check 
        #Returns index of the shortest path 
        return index

    # To check for visited nodes for the A* algorithm
    def check_visited(self,current,visited):
        return current in visited

    #A* algorithm to find the shortest path from the start orientation to goal orientation
    def A_star(self):
        # List of nodes to be evaluated
        track = []
        path_found = False
        # List of nodes already visited
        visited = []
        # Starting point of the vehicle
        start = ego_veh_start
        tot_cost = 0   
        g_cost = 0   
        path = [start]
        # Add the starting node to the list of nodes to be evaluated
        track.append((start,tot_cost,g_cost,path))  
        # Run the algorithm as long as there are nodes to be evaluated 
        while len(track)>0:
            # Get the index of the shortest path from the list of nodes to be evaluated
            index = self.priority(track)
            (shortest,_,g_val,path) = track[index] 
            track.pop(index)
            # Check if the current node has not been visited before
            if not (self.check_visited([round(shortest[0]),round(shortest[1]),round(shortest[2])],visited)): 
                visited.append([round(shortest[0]),round(shortest[1]),round(shortest[2])])
                # Check if the current node is the goal node
                if round(shortest[0]) <= ego_veh_goal[0]+(self.robot.extinguish_rad/sc_ratio) and round(shortest[0]) >= ego_veh_goal[0]-(self.robot.extinguish_rad/sc_ratio) and round(shortest[1]) <= ego_veh_goal[1]+(self.robot.extinguish_rad/sc_ratio) and round(shortest[1]) >= ego_veh_goal[1]-(self.robot.extinguish_rad/sc_ratio) : 
                    #If the goal is reached
                    path_found= True
                    return path, path_found
                # Valid neighbours following kinematic equations and non holonomic constraints
                # Get the neighbors of the current node
                neighbours= self.get_neighbours(shortest[0],shortest[1],shortest[2]) 
                
                # Evaluate the neighbors of the current node
                for neighbour in neighbours:
                    vel = neighbour[3]
                    turn = neighbour[4]
                    new_g_cost = g_val+(self.dist_cost(shortest[0],shortest[1],neighbour[0],neighbour[1]))
                    new_tot_cost = new_g_cost+(self.heuristic_function(neighbour[0],neighbour[1],turn,shortest[4],vel,shortest[3]))
                    if not (self.check_visited([round(neighbour[0]),round(neighbour[1]),round(neighbour[2])],visited)):
                        track.append((neighbour,new_tot_cost,new_g_cost,path+[neighbour]))
        print("A star couldnt find a path")      
        return path, path_found
    
    #Random Position to Spawn the Truck 
    def spawn_random(self):
     while True:
         x = random.uniform(0, self.world.width)
         y = random.uniform(0, self.world.height)
         theta = random.uniform(0, 360)
         if self.valid_point(x, y, theta):
             return [x, y, theta, 0, 0]


class Simulation:

    def __init__(self,robot = Robot_Kinematics , world= World, controller = Planner_Controller) -> None:
        self.robot = robot
        self.world = world
        self.controller = controller

    #Wumpus Loop to set Obstacles on Fire
    def wumpus_burn_mod_old(self):
        valid_burn_indices = []
        for i in range(len(self.world.obstacle_state)):
            if self.world.obstacle_state[i][2] != 'B':
                if not any('B' in self.world.obstacle_state[max(i-15, 0):min(i+15, len(self.world.obstacle_state))][j][2] for j in range(len(self.world.obstacle_state[max(i-15, 0):min(i+15, len(self.world.obstacle_state))]))):
                    valid_burn_indices.append(i)
        if len(valid_burn_indices) > 0:
            burn_index = random.choice(valid_burn_indices)
            self.world.obstacle_state[burn_index][2] = 'B'
            self.world.burning_list.append([self.world.obstacle_state[burn_index][0], self.world.obstacle_state[burn_index][1]])
            print("Wumpus: Setting another obstacle on fire XD >< ><")  
        threading.Timer(15, self.wumpus_burn_mod_old).start()
        return
    
    def extinguish_check(self,x,y):
        e_r = 40
        for x_, y_ in self.world.burning_list:
            if abs(x - x_) <= round(e_r / sc_ratio) and abs(y - y_) <= round(e_r / sc_ratio):
                return True
            else:
                return False
            
    #If not extinguished spread fire nearby to all obstacles
    def burn_spread(self):
        new_burn = []
        for i, obs1 in enumerate(self.world.obstacle_state):
            if obs1[2] == 'B':
                for j, obs2 in enumerate(self.world.obstacle_state[i+1:], start=i+1):
                    if abs(obs2[0] - obs1[0]) < (self.robot.burn_rad) and abs(obs2[1] - obs1[1]) < (self.robot.burn_rad) and obs2[2] != 'B':
                        new_burn.append(j)
                        obs2[2] = 'B'
                        new_burning_location = [obs2[0], obs2[1]]
                        if new_burning_location not in self.world.burning_list:
                            self.world.burning_list.append(new_burning_location)
                            print(" ALERT :: Fire is Spreading!")
        threading.Timer(20, self.burn_spread).start()
        return

    # Extinguish the burning obstacles if the Firetruck is near 
    def extinguish_burn(self, x, y):
        global cnt_extinguish
        new_burning_list = []
        for x_, y_ in self.world.burning_list:
            if abs(x - x_) <= round(self.robot.extinguish_rad / sc_ratio) and abs(y - y_) <= round(self.robot.extinguish_rad / sc_ratio):
                cnt_extinguish += 1
            else:
                new_burning_list.append([x_, y_])

        self.world.burning_list = new_burning_list
        self.state_change()
        return
    
    #Change the state of obstacles that are extinguished
    def state_change(self):
        for obstacle in self.world.obstacle_state:
            if obstacle[2] == 'B' and [obstacle[0], obstacle[1]] not in self.world.burning_list:
                obstacle[2] = 'N'

    # Main simulation animation loop 
    def animate(self, pos_x, pos_y, theta):
        # Extinguish the burning obstacles
        self.extinguish_burn(round(pos_x), round(pos_y))
        if not self.extinguish_check(round(pos_x), round(pos_y)):
            for x in range(2,6,2):
                self. extinguish_burn(round(pos_x+x), round(pos_y+x))
                self. extinguish_burn(round(pos_x-x), round(pos_y-x))
        
        plt.figure(self.world.coverage)
        fig = plt.gcf()
        ax = fig.add_subplot(1, 1, 1)
        fig.canvas.manager.set_window_title("Search Based Planning")
        track = 0
        for x in range(self.world.width):
            for y in range(self.world.height):
                if self.world.grid[x][y] == 1:
                    if self.world.obstacle_state[track][2] == 'N':
                        plt.scatter(x, y, c='green', s=3, marker="s")
                    elif self.world.obstacle_state[track][2] == 'B':
                        plt.scatter(x, y, c='red', s=6, marker="s")
                    track += 1

        plt.xlim([-1, self.world.width])
        plt.ylim([-1, self.world.height])
        plt.axis('off')

        ego_veh = mpimg.imread(ego_veh_path)
        """ To visulise rotation property as well uncomment the next two lines"""
        #imagebox_4 = self.rotate_bound(ego_veh, -theta)
        #imagebox_4 = OffsetImage(imagebox_4, zoom=0.125)
        """ Comment this line and uncomment above """
        imagebox = OffsetImage(ego_veh, zoom=0.125)
        ab = AnnotationBbox(imagebox, (pos_x, pos_y), bboxprops=dict(edgecolor='white'))
        ax.add_artist(ab)

        return
    
    def rotate_bound(self,image, angle):
        # Get the dimensions of the image and determine the center
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get the rotation matrix and calculate the sine and cosine
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        # Calculate the new bounding dimensions of the image
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust the rotation matrix to take into account translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform the rotation and return the image
        return cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))

    # To find the nearest burning obstacle
    def get_nearest(self):
        nearest_burning = None
        min_distance = math.inf
    
        for x, y in self.world.burning_list:
            distance = math.sqrt((ego_veh_start[0] - x) ** 2 + (ego_veh_start[1] - y) ** 2)
            if distance < min_distance and (ego_veh_goal[0], ego_veh_goal[1]) != (x, y):
                min_distance = distance
                nearest_burning = (x, y)
        
        if nearest_burning is None:
            return True, ego_veh_goal
        else:
            return False, nearest_burning

    # Gets random burning obstacle index 
    def get_next(self):
        if len(self.world.burning_list) > 0:
            random_idx = random.randint(0, len(self.world.burning_list)-1)
            x, y = self.world.burning_list[random_idx]
            return False, (x, y)
        else:
            return True, ego_veh_goal

    def simulation_begin(self):
        global sim_end
        if sim_end:
            sim_end =False
            threading.Timer( 3*60, self.simulation_begin).start()
        else:
            sim_end = True
        return

    def get_intact(self):
        intact=0
        for obs in self.world.obstacle_state:
            if obs[2]=='N':
                intact+=1
        return intact

    def get_burned(self):
        burned=0
        for obs in self.world.obstacle_state:
            if obs[2]!='N':
                burned+=1
        return burned


    
def main():
    global ego_veh_goal
    global ego_veh_start
    global sim_end
    global cnt_extinguish
    global cpu_time

    world = World()
    robot = Robot_Kinematics()
    controller = Planner_Controller(robot,world)
    sim = Simulation(robot,world,controller)
    # Create the grid with desired coverage
    world.create_grid(world.coverage)
    # Spawn the fire truck on the grid
    ego_veh_start = controller.spawn_random()
    # Begin Simulation 
    sim.simulation_begin()
    # Wumpus Start burn 
    sim.wumpus_burn_mod_old()
    threading.Timer(20, sim.burn_spread).start()
    plt.cla()

    sim.animate(ego_veh_start[0],ego_veh_start[1],ego_veh_start[2])
    plt.pause(0.001)

    while not sim_end:
        # Get the nearest point to the fire truck and wait until it's safe to move
        wait_state,ego_veh_goal = sim.get_nearest()
        if wait_state:
            continue
        t1 = time.time()
        # Run A* algorithm to find the path to the goal
        path, path_found =controller.A_star()
        cpu_time+=(time.time()-t1)
        for points in path:
            plt.cla()
            # Animate the simulation with each point in the path
            sim.animate(points[0],points[1],points[2])
            plt.pause(0.00001)
            plt.axis('off')
        # Update the start position of the fire truck to the end of the path
        ego_veh_start=[path[-1][0],path[-1][1],path[-1][2],0,0]
        print("I reached to save you all! God is here!")
    threading.Timer( 20, sim.burn_spread).cancel()
    threading.Timer( 61, sim.wumpus_burn_mod_old).cancel()
    plt.close()
    print("Program Ended")
    print("Intact "+str(sim.get_intact()))
    print("Burned "+str(sim.get_burned()))
    print("Cnt_extinguished "+str(cnt_extinguish))
    print("Total CPU time "+str(cpu_time))
    print("The program is over")
    plt.close()
    

if __name__ == "__main__":
    main()