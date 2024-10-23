import sys

import numpy as np
import math
import os
import json
import random
import hashlib
import matplotlib.pyplot as plt

from vis_nav_core import Game
import pub as nopub

class Maze3DGenerator:
    def __init__(self, maze_input):
        self.selected_textures = []
        self.textures_path = os.path.join('.', 'textures')
        self.enable_furniture = 1
        self.panel_size = 0.2

        self.save_dir = "data/"
        self.saved_panel_data = []
        self.saved_maze_data = []

        self.maze = maze_input

        #For target scene location
        self.target_pose = [0, 0, 0]
        self.panel_size = 0.2

        #Manually set FPV robot pose
        self.robot_x_pose = 0.2
        self.robot_y_pose = 0.2

        self.maze_max_x = maze_input.shape[0] * self.panel_size
        self.maze_max_y = maze_input.shape[1] * self.panel_size

        self.constraint_dist_min = 6
        self.constraint_dist_max = 12

        self.constraint_to_x = 0.1
        self.constraint_to_y = 0.1

    def meet_constraint(self, target_pose, reference_pose):
        is_good = -1
        actual_dist = np.sqrt((target_pose[0] - reference_pose[0]) ** 2 + (target_pose[1] - reference_pose[1]) ** 2)

        if self.constraint_dist_min < actual_dist < self.constraint_dist_max:
            is_good = 1

        val_x = (abs(target_pose[0] - self.maze_max_x) < self.constraint_to_x) or (
                    abs(target_pose[0]) < self.constraint_to_x)
        val_y = (abs(target_pose[1] - self.maze_max_y) < self.constraint_to_y) or (
                    abs(target_pose[1]) < self.constraint_to_y)
        if val_x or val_y:
            is_good = -1

        return is_good

    def random_sample(self):
        selected_target_x = selected_target_y = 0.0

        # target_yaw = random.uniform(-math.pi/2, math.pi/2)

        while True:
            # select a random matrix entry with value==0
            rows, cols = np.where(self.maze == 0)
            nth = np.random.randint(len(rows))
            selected_pose = rows[nth], cols[nth]

            print("selected pose: ", selected_pose)
            selected_target_x = round(selected_pose[0] * self.panel_size, 3)
            selected_target_y = round(selected_pose[1] * self.panel_size, 3)

            # print(f"comparing: [{selected_target_x}, {selected_target_y}],[{self.reset_x_pose}, {self.reset_y_pose}]")
            if self.meet_constraint([selected_target_x, selected_target_y],
                                    [self.robot_x_pose, self.robot_y_pose]) == 1:
                break

        self.target_pose = [selected_target_x, selected_target_y, 0.01]
        print(f"Target Pose to origin: [{selected_target_x},{selected_target_y},0.01]")
        maze = np.array(self.maze)
        maze[selected_pose] = 2
        np.set_printoptions(threshold=sys.maxsize)
        print(np.array2string(maze, separator=' '))
    
    def maze_generator(self):
        '''
        define the cube's size
        create the cube for the maze.world file from the structure of the maze input 
        '''
        x, y, z, roll, pitch, yaw = 0,0,0,0,0,0

        model_type = 0

        self.saved_maze_data.append(self.maze.tolist())

        new_maze = self.regenerate_matrix(self.maze)

        for i in range(len(new_maze)): # loop the maze's row
            for j in range(len(new_maze[0])): #loop the maze's col
                if new_maze[i,j] != 0:
                    #reset value
                    x, y, z, roll, pitch, yaw= 0,0,0,0,0,0

                    # model_type also is the integer part of the floating number
                    model_type = math.floor(new_maze[i,j]) 
                    # extract the fraction part of the floating number
                    floating_part = (new_maze[i,j]*10)%10

                    # meet model_1
                    if model_type == 1:
                        # offset on this object
                        roll = 0.0

                        if floating_part == 1:
                            yaw = 0.0

                        elif floating_part == 2:
                            yaw = math.pi/2.0
                    
                        else:
                            yaw = 0.0

                    # meet model_2
                    elif model_type == 2:
                        #bottom left
                        if floating_part == 1:
                            yaw = math.pi/2.0

                        #top right
                        elif floating_part == 2:
                            yaw = -1*math.pi/2.0

                        #top left
                        elif floating_part == 3:
                            yaw = 0.0
                        
                        #bottom right
                        elif floating_part == 4:
                            yaw = math.pi

                    # meet model_3
                    elif model_type == 3:
                        #left side
                        if floating_part == 1:
                            yaw = math.pi/2

                        #bottom side
                        elif floating_part == 2:
                            yaw = math.pi
                        
                        #right side
                        elif floating_part == 3:
                            yaw = -1*math.pi/2
                        
                        #upper side
                        elif floating_part == 4:
                            yaw = 0.0

                    # meet model_4
                    elif model_type == 4:
                        pass

                    #Enable different types of model
                    if model_type == 1 or model_type == 2 or model_type == 3 or model_type == 4: 
                        x = self.panel_size * j 
                        y = self.panel_size * i 

                        selected_texture_index = self.get_random_texture()

                        self.saved_panel_data.append([model_type, selected_texture_index, x,y,0, roll,pitch,yaw])

    ## Remake matrix
    def find_indices(self, array, value):
        indices = []

        for index, element in np.ndenumerate(array):
            if element == value:
                indices.append(list(index))

        return indices

    def regenerate_matrix(self, multi_array):
        multi_array_copy = np.array(multi_array)

        indices = self.find_indices(multi_array, 1)
        new_array = np.zeros_like(multi_array_copy, dtype=float)

        for idx, idy in indices:
            #Generate left, right, up, and down value
            left_i , left_j = idx, idy-1
            right_i, right_j = idx, idy+1
            up_i, up_j = idx-1, idy
            down_i, down_j = idx+1, idy
            
            left_val, right_val, up_val, down_val = 0, 0, 0, 0

            #check whether index is valid and check whether the value at this location is 1 or 0
            if (left_i >= 0 and left_i < multi_array_copy.shape[0]) and (left_j >= 0 and left_j < multi_array_copy.shape[1]):
                if multi_array_copy[left_i, left_j] == 1:
                    left_val = 1

            if (right_i >= 0 and right_i < multi_array_copy.shape[0]) and (right_j >= 0 and right_j < multi_array_copy.shape[1]):
                if multi_array_copy[right_i, right_j] == 1:
                    right_val = 1

            if (up_i >= 0 and up_i < multi_array_copy.shape[0]) and (up_j >= 0 and up_j < multi_array_copy.shape[1]):
                if multi_array_copy[up_i, up_j] == 1:
                    up_val = 1

            if (down_i >= 0 and down_i < multi_array_copy.shape[0]) and (down_j >= 0 and down_j < multi_array_copy.shape[1]):
                if multi_array_copy[down_i, down_j] == 1:
                    down_val = 1

            res = left_val + right_val + up_val + down_val
            
            # Detect whether this index is on the side
            if (left_val and right_val and ((up_val+down_val)== 0)) or (up_val and down_val and ((left_val+right_val)== 0)):
                res -= 1
            
            # Special case that no objects are surrounding
            if multi_array_copy[idx, idy] == 1 and res == 0:
                res = 1

            # Determine the orientation of model 1
            if res == 1:
                # model_1 in 0 deg rotation
                # ''
                if ((left_val or right_val) and ((up_val+down_val)== 0)):
                    res = 1.1
                # model_1 in 90 deg rotation
                # ' 
                # '
                elif ((up_val or down_val) and ((left_val+right_val)== 0)):
                    res = 1.2

                else:
                    res = 1.3

            #determine the orientation of model 2    
            elif res == 2:
                #'
                #'''
                if (right_val and down_val and ((left_val+up_val)== 0)):
                    res = 2.1
                #'''
                #  '
                elif (left_val and down_val and ((right_val+up_val)== 0)):
                    res = 2.4
                #'''
                #'
                elif (right_val and up_val  and ((left_val+down_val)== 0)):
                    res = 2.3
                #   '
                # '''
                elif (left_val and up_val  and ((right_val+down_val)== 0)):
                    res = 2.2

            #determine the orientation of model 3
            elif res == 3:
                #'
                #'''
                #'
                if (right_val and up_val and down_val and (left_val == 0)):
                    res = 3.1
                #  '
                # '''
                elif (left_val and right_val and down_val and (up_val == 0)):
                    res = 3.2
                #  '
                #'''
                #  '
                elif (left_val and up_val  and down_val and (right_val == 0)):
                    res = 3.3
                #'''
                # '
                elif (left_val and right_val and up_val and (down_val == 0)):
                    res = 3.4

            elif res == 4:
                  res = 4.1

            # print(res)
            new_array[idx, idy] = res
            
        return new_array
    
    def get_random_texture(self):
        select_num = random.randint(1, 200)
        # select_num = -1
        # while True:
        #     pick = random.randint(1, 200)
        #     if pick not in self.selected_textures:
        #         self.selected_textures.append(pick)
        #         select_num = pick
        #         # print(len(self.selected_textures),": select: ", select_num)
        #         break

        return select_num

    def save_data_to_disk(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        panel_array = np.array(self.saved_panel_data).tolist()
        target_array = np.array(self.target_pose).tolist()
        panel_json_array = json.dumps(panel_array)
        target_json_array = json.dumps(target_array)

        panel_encrypted_data = nopub.encrypt_data(panel_json_array.encode('utf-8'))
        target_encrypted_data = nopub.encrypt_data(target_json_array.encode('utf-8'))

        import time
        time_str = input('When should Navigation phase start (YYYY/mm/dd HH:MM:SS)?')
        NAV_START_TIME = time.mktime(time.strptime(time_str, '%Y/%m/%d %H:%M:%S'))
        NOISY_MOTION = 0 if '0' == input('Enable noisy motion? (1 - Yes, 0 - No)?') else 1
        rules_dict = {
            'NAV_START_TIME': NAV_START_TIME,
            'NOISY_MOTION': NOISY_MOTION
        }
        rules_str = json.dumps(rules_dict)
        rules_byte = rules_str.encode('utf-8')
        rules_byte_en = nopub.encrypt_data(rules_byte)

        combined_data = {
            "panel_data": panel_encrypted_data.decode('utf-8'),
            "target_data": target_encrypted_data.decode('utf-8')
        }

        combined_data_with_rule = {
            "panel_data": panel_encrypted_data.decode('utf-8'),
            "target_data": target_encrypted_data.decode('utf-8'),
            "rules_data": rules_byte_en.decode('utf-8')
        }

        combined_json = json.dumps(combined_data)
        combined_json_with_rule = json.dumps(combined_data_with_rule)
        
        # Save the JSON data to a file
        file_path = os.path.join(self.save_dir, "maze.json")
        self.save_file_and_md5(combined_json, file_path)

        file_path = os.path.join(self.save_dir, "maze-with-rules.json")
        self.save_file_and_md5(combined_json_with_rule, file_path)

        print("Finished saving the maze and target info. as files to the disk")

    def save_file_and_md5(self, data_str, file_path):
        with open(file_path, 'w') as file:
            file.write(data_str)

        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as file:
            for chunk in iter(lambda: file.read(4096), b''):
                md5_hash.update(chunk)
        
        print(f"MD5 hash of '{file_path}': {md5_hash.hexdigest()}")

        output_file = file_path+'-md5.txt'
        with open(output_file, 'w') as f:
            f.write(md5_hash.hexdigest())

        return md5_hash.hexdigest()
    
class Maze:
    def __init__(self, num_rows, num_cols):
        #this robot id will be used to generate the target scene
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.max_room_size=3
        self.room_density=0.1

    def generate_maze(self, length, width):
        # initialize the maze with all walls
        maze = np.ones((length, width), dtype=int)

        # create the outer perimeter of wall cells
        maze[0,:] = 1; maze[-1,:] = 1; maze[:,0] = 1; maze[:,-1] = 1

        # set the top-left cell as the starting point
        maze[1,1] = 0

        # perform a randomized depth-first search to create paths in the maze
        stack = [(1, 1)]
        while stack:
            current_row, current_col = stack.pop()
            neighbors = self.get_unvisited_neighbors(current_row, current_col, maze)

            if neighbors:
                # choose a random neighbor to visit
                neighbor_row, neighbor_col = neighbors[np.random.randint(len(neighbors))]
                # carve a path to the neighbor
                maze[current_row + (neighbor_row - current_row)//2, current_col + (neighbor_col - current_col)//2] = 0
                maze[neighbor_row, neighbor_col] = 0
                stack.append((current_row, current_col))
                stack.append((neighbor_row, neighbor_col))

        # randomly place rooms in the maze
        for i in range(2, length-2):
            for j in range(2, width-2):
                if maze[i, j] == 0 and np.random.rand() < self.room_density:
                    room_length = np.random.randint(1, self.max_room_size+1)
                    room_width = np.random.randint(1, self.max_room_size+1)
                    if i + room_length < length and j + room_width < width:
                        maze[i:i+room_length, j:j+room_width] = 0

        return maze

    def get_unvisited_neighbors(self, row, col, maze):
        neighbors = []
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        for direction in directions:
            neighbor_row = row + direction[0]
            neighbor_col = col + direction[1]
            if (0 <= neighbor_row < maze.shape[0] and
                0 <= neighbor_col < maze.shape[1] and
                maze[neighbor_row, neighbor_col] == 1):
                neighbors.append((neighbor_row, neighbor_col))
        return neighbors

    def visualize_maze(self, maze):
        # create a figure and axis object
        fig, ax = plt.subplots()

        # create a color map for the walls and the paths
        cmap = plt.get_cmap('binary', 2)

        # plot the maze as an image
        ax.imshow(maze, cmap=cmap, interpolation='nearest')

        # set the ticks to show only at integer values and label them
        ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
        ax.set_xticklabels(np.arange(0, maze.shape[1]+1, 1))
        ax.set_yticklabels(np.arange(0, maze.shape[0]+1, 1))
        # set the axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # set the grid lines to be white and only show for integer values
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

        # set the axis limits to show the whole maze
        ax.set_xlim([-0.5, maze.shape[1]-0.5])
        ax.set_ylim([-0.5, maze.shape[0]-0.5])

        # set the axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.axis('off')

        # show the plot
        plt.show()

    def run(self):
        # generate the maze
        print('Creating Maze: ', self.num_rows, "x", self.num_cols)
        maze = self.generate_maze(self.num_rows, self.num_cols)
        print(maze)
    
        maze_3D = Maze3DGenerator(maze)
        # Generate maze and panels array
        maze_3D.maze_generator()

        # Generate target location
        maze_3D.random_sample()

        # write maze, panels and target scene location to the disk
        maze_3D.save_data_to_disk()



if __name__ == "__main__":
    maze_3D = Maze(31, 31)
    maze_3D.run()
    