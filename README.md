# Wildfire

The task is to develop planners that employ various motion planning techniques for guiding a firetruck and its hostile Wumpus adversary through a maze-like environment cluttered with obstacles.

### Environment Setup
The two-dimensional 250X250 grid size has the following properties: 

Multiple randomly placed obstacles in the environment  that are made up of extensive areas of thick vegetation, trees, and weeds that are shaped like enormous tetrominoes, making them difficult to navigate.The percentage of coverage of the obstacles is 15 percent.


The main characters of the Wildfire Simulation are:

A] Obstacles on the Environment Grid

The environment world in which the firetruck and Wumpus navigate consists of obstacles arranged randomly in the shape of tetris blocks on a grid. The grid density for the simulation is set at 15 percent, and each obstacle square has a base dimension of 5 meters. The obstacles are initially green in color, resembling trees and wild bushes in a forest. The program maintains a list that updates the state of all obstacles throughout the simulation. The obstacles on the grid are classified into two states: Burning and Non-Burning. The burning state is indicated by the letter “B”, representing the obstacles set on fire by the Wumpus, and these obstacles are visualized in red. All other obstacles are considered non-burning and “NB” and appear green on the grid.
 
B] The Wumpus 

In this world, the culprit who sets fire to obstacles is a Wumpus that adheres to standard issue characteristics. Besides its distinctive odor, it also possesses the ability to ignite fires in obstacles situated in its proximity. As its motion is restricted to a grid, it necessitates a combinatorial motion planning algorithm to navigate. Specifically, the Wumpus employs the Hybrid A* algorithm for implementation. Initially, the Wumpus begins in a randomly assigned position on the grid and awaits its path planner to chart a course to a random obstacle to ignite. After the path planner returns the path, the Wumpus proceeds to follow the directions, setting fire to obstacles. It continues roaming around, igniting random obstacles, while the path planner works in the background to ensure proper guidance.


C] The Fire Truck 

The fire truck is a robot designed in the form of a car and constrained by the non-holonomic constraints of a car, with Ackerman steering based on fixed vehicle parameters. The robot is supported by a drone that provides it with omniscient perception, enabling it to detect the location of the nearest fire within the truck's vicinity. The robot then follows a path planner to reach the desired goal location. To plan its path, the truck employs a sampling-based planner that uses a probabilistic roadmap and a local planner to guide its path to extinguish the fire and save the environment. 

<p align = "center"> <img width="575" alt="Screenshot 2023-05-02 at 5 45 05 AM" src="https://user-images.githubusercontent.com/115124698/235553333-bf296a46-6e5c-4e63-a186-4e0b8f3114b7.png">


### Implementation Details 
To comprehensively analyze the performance of both combinatorial and sampling-based planning methods for navigating a fire truck through a deadly obstacle field in an attempt to extinguish as many fires as possible, three separate visualization programs have been implemented. 

A] Search Based Sampling Implementation

 In this, the fire truck uses a search-based planner, specifically a hybrid A* algorithm, to plan a path from its current location to the nearest burning location.
The implementation involves simulating a grid with random obstacles placed at desired percentage coverage in the forest. Each obstacle is of a fixed size, and its burning status is continuously monitored throughout the simulation. The Wumpus is also introduced and set loose who  randomly selects obstacles to set on fire at fixed intervals.
The fire truck then implements the search-based algorithm A* hybrid variant to find a path from its current position to the nearest burning obstacle. When the truck reaches a distance of 5 meters from the obstacle and stays there for 5 seconds, the obstacle is extinguished.

The fire truck then waits for the next location of the nearest burning obstacle and moves to extinguish the fire. This process continues until the simulation time is over.

B] Sampling Based Implementation 

 The program utilizes a sampling-based planner, specifically a Probabilistic Roadmap (PRM) that incorporates the A* algorithm as its local planner, to chart the optimal path from the current position of the fire truck to the closest burning location. To simulate a forest with obstacles, a grid is created with randomly distributed obstacles at predetermined coverage percentages. These obstacles are of a fixed size, and their status is continuously monitored during the simulation. Additionally, a Wumpus is introduced and set loose to randomly select obstacles to ignite at fixed intervals. 
The first step in the implementation process is generating a PRM for a 2D grid. The roadmap is constructed by sampling random points on the grid and connecting them if there are no obstacles between them. The fire truck is then initialized at random points, and if those points are not present on the grid, they are connected to the nearest existing node on the graph using the connect_node function.

The local path planner utilizes the A* algorithm, a search-based algorithm, to determine the shortest path between waypoints until the truck reaches its goal. Once the truck is within 5 meters of an obstacle, it remains stationary for 5 seconds to extinguish the fire before moving on to the next burning location. The fire truck then waits for the next location of the nearest burning obstacle and moves to extinguish the fire. This process continues until the simulation time ends.

C] Integrated Path Planning Based Implementation

This implementation involves an integrated combinational approach, whereby the Wumpus uses a Search-Based path planner to light obstacles on fire, and the fire truck employs a probabilistic roadmap planner to chart its path to extinguish the fires. To simulate a forest with obstacles, a grid is generated with obstacles randomly distributed at predetermined coverage percentages. The obstacles are fixed in size, and their status is continuously monitored during the simulation.

Specifically, the Wumpus uses the Hybrid A* algorithm to implement its role. It begins in a randomly assigned position on the grid and awaits its path planner to chart a course to a random obstacle to ignite. After the path planner returns the path, the Wumpus proceeds to follow the directions, setting fire to obstacles. The Wumpus continues roaming around, igniting random obstacles, while the path planner works in the background to ensure proper guidance. This process operates on a separate thread, and the Wumpus is never tired from running around and lighting trees on fire.Even though Wumpus is not visualized, it is working in the background to lit obstacles on fire 

The implementation process for Fire Truck begins by generating a PRM for a 2D grid. The roadmap is constructed by randomly sampling points on the grid and connecting them if there are no obstacles between them. The fire truck is initialized at random points, and if those points are not present on the grid, they are connected to the nearest existing node on the graph using the connect_node function.

The local path planner employed by the fire truck uses the A* algorithm, a search-based algorithm, to determine the shortest path between waypoints until the truck reaches its goal. When the truck is within 5 meters of an obstacle, it remains stationary for 5 seconds to extinguish the fire before moving on to the next burning location. The fire truck then awaits the next location of the nearest burning obstacle and moves to extinguish the fire. This process continues until the simulation time ends.

