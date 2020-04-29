from environments.warehouse.item import Item
from environments.warehouse.robot import Robot
from environments.warehouse.utils import *
import numpy as np
import copy
import random
from gym import spaces
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import csv

class Warehouse(object):
    """
    warehouse environment
    """

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}

    def __init__(self, parameters:dict={}):
        parameters = read_parameters('warehouse')
        # parameters = parse_arguments()
        self.n_columns = parameters['n_columns']
        self.n_rows = parameters['n_rows']
        self.n_robots_row = parameters['n_robots_row']
        self.n_robots_column = parameters['n_robots_column']
        self.distance_between_shelves = parameters['distance_between_shelves']
        self.robot_domain_size = parameters['robot_domain_size']
        self.prob_item_appears = parameters['prob_item_appears']
        # The learning robot
        self.learning_robot_id = parameters['learning_robot_id']
        self.n_steps_episode = parameters['n_steps_episode']
        self.obs_type = parameters['obs_type']
        self.items = []
        self.img = None
        self.log_obs = parameters['log_obs']
        self.log_file = parameters['log_file']
        self.reset()

    ############################## Override ###############################

    def reset(self):
        """
        Resets the environment's state
        """
        self.robot_id = 0
        self._place_robots()
        self.item_id = 0
        self.items = []
        self._add_items()
        obs = self._get_observation()
        self.num_steps = 0
        return obs

    def step(self, action):
        """
        Performs a single step in the environment.
        """
        if self.log_obs is True:
            self._log_obs(self.log_file, action)
        actions = []
        for robot in self.robots:
            state = self._get_state()
            obs = robot.observe(state, self.obs_type)
            actions.append(robot.select_random_action())
        actions[self.learning_robot_id] = action
        self._robots_act(actions)
        self._increase_item_waiting_time()
        reward = self._compute_reward(self.robots[self.learning_robot_id])
        self._remove_items()
        self._add_items()
        obs = self._get_observation()
        # Check whether learning robot is done
        # done = self.robots[self.learning_robot_id].done
        self.num_steps += 1
        done = (self.n_steps_episode <= self.num_steps)
        # Experiment.py resets the environment when done
        # if done is True:
        #     # Reset the environment to start a new episode.
        #     self.reset()
        return obs, reward, done, []

    @property
    def observation_space(self):
        return None

    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        n_actions = spaces.Discrete(len(self.ACTIONS))
        action_dict = {robot.get_id:n_actions for robot in self.robots}
        action_space = spaces.Dict(action_dict)
        action_space.n = 4
        return action_space

    def render(self, delay=0.0):
        """
        Renders the environment
        """
        bitmap = self._get_state()
        position = self.robots[self.learning_robot_id].get_position
        bitmap[position[0], position[1], 1] += 1
        im = bitmap[:, :, 0] - 2*bitmap[:, :, 1]
        if self.img is None:
            fig,ax = plt.subplots(1)
            self.img = ax.imshow(im)
            for robot_id, robot in enumerate(self.robots):
                domain = robot.get_domain
                y = domain[0]
                x = domain[1]
                if robot_id == self.learning_robot_id:
                    color = 'r'
                    linestyle='-'
                    linewidth=2
                else:
                    color = 'k'
                    linestyle=':'
                    linewidth=1
                rect = patches.Rectangle((x-0.5, y-0.5), self.robot_domain_size[0],
                                         self.robot_domain_size[1], linewidth=linewidth,
                                         edgecolor=color, linestyle=linestyle,
                                         facecolor='none')
                ax.add_patch(rect)
        else:
            self.img.set_data(im)
        plt.pause(delay)
        plt.draw()

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def create_graph(self, robot):
        """
        Creates a graph of robot's domain in the warehouse. Nodes are cells in
        the robot's domain and edges represent the possible transitions.
        """
        graph = nx.Graph()
        for i in range(robot.get_domain[0], robot.get_domain[2]+1):
            for j in range(robot.get_domain[1], robot.get_domain[3]+1):
                cell = np.array([i, j])
                graph.add_node(tuple(cell))
                for neighbor in self._neighbors(cell):
                    graph.add_edge(tuple(cell), tuple(neighbor))
        return graph

    ######################### Private Functions ###########################

    def _place_robots(self):
        """
        Sets robots initial position at the begining of every episode
        """
        self.robots = []
        domain_rows = np.arange(0, self.n_rows, self.robot_domain_size[0]-1)
        domain_columns = np.arange(0, self.n_columns, self.robot_domain_size[1]-1)
        for i in range(self.n_robots_row):
            for j in range(self.n_robots_column):
                robot_domain = [domain_rows[i], domain_columns[j],
                                domain_rows[i+1], domain_columns[j+1]]
                robot_position = [robot_domain[0] + self.robot_domain_size[0]//2,
                                  robot_domain[1] + self.robot_domain_size[1]//2]
                self.robots.append(Robot(self.robot_id, robot_position,
                                                  robot_domain))
                self.robot_id += 1

    def _add_items(self):
        """
        Add new items to the designated locations in the environment which
        need to be collected by the robots
        """
        item_columns = np.arange(0, self.n_columns)
        item_rows = np.arange(0, self.n_rows, self.distance_between_shelves)
        item_locs = None
        if len(self.items) > 0:
            item_locs = [item.get_position for item in self.items]
        for row in item_rows:
            for column in item_columns:
                loc = [row, column]
                loc_free = True
                if item_locs is not None:
                    loc_free = loc not in item_locs
                if random.random() < self.prob_item_appears and loc_free:
                    self.items.append(Item(self.item_id, loc))
                    self.item_id += 1
        item_rows = np.arange(0, self.n_rows)
        item_columns = np.arange(0, self.n_columns, self.distance_between_shelves)
        if len(self.items) > 0:
            item_locs = [item.get_position for item in self.items]
        for row in item_rows:
            for column in item_columns:
                loc = [row, column]
                loc_free = True
                if item_locs is not None:
                    loc_free = loc not in item_locs
                if random.random() < self.prob_item_appears and loc_free:
                    self.items.append(Item(self.item_id, loc))
                    self.item_id += 1


    def _get_state(self):
        """
        Generates a 3D bitmap: First layer shows the location of every item.
        Second layer shows the location of the robots.
        """
        state_bitmap = np.zeros([self.n_rows, self.n_columns, 2], dtype=np.int)
        for item in self.items:
            item_pos = item.get_position
            state_bitmap[item_pos[0], item_pos[1], 0] = 1 #item.get_waiting_time
        for robot in self.robots:
            robot_pos = robot.get_position
            state_bitmap[robot_pos[0], robot_pos[1], 1] = 1
        return state_bitmap

    def _get_observation(self):
        """
        Generates the individual observation for every robot given the current
        state and the robot's designated domain.
        """
        state = self._get_state()
        observation = self.robots[self.learning_robot_id].observe(state, self.obs_type)
        shape = np.shape(observation)
        observation = np.reshape(observation, (shape[0], shape[1], 1))
        return observation

    def _robots_act(self, actions):
        """
        All robots take an action in the environment.
        """
        for action,robot in zip(actions, self.robots):
            robot.act(action)

    def _compute_reward(self, robot):
        """
        Computes reward for the learning robot.
        """
        reward = 0
        robot_pos = robot.get_position
        robot_domain = robot.get_domain
        for item in self.items:
            item_pos = item.get_position
            if robot_domain[0] <= item_pos[0] <= robot_domain[2] and \
               robot_domain[1] <= item_pos[1] <= robot_domain[3]:
                reward += -0.1 #*item.get_waiting_time
            if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                reward += 1
        return reward


    def _remove_items(self):
        """
        Removes items collected by robots. Robots collect items by steping on
        them
        """
        for robot in self.robots:
            robot_pos = robot.get_position
            for item in self.items:
                item_pos = item.get_position
                if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                    self.items.remove(item)

    def _increase_item_waiting_time(self):
        """
        Increases items waiting time
        """
        for item in self.items:
            item.increase_waiting_time()

    def _neighbors(self, cell):
        return [cell + [0, 1], cell + [0, -1], cell + [1, 0], cell + [-1, 0]]

    def _log_obs(self, log_file, action):
        """
        Logs observations into a csv file
        """
        pass
        # with open (log_file,'a') as file:
        #     robot = self.robots[self.learning_robot_id]
        #     robot_domain = robot.get_domain
        #     state = self._get_state()
        #     items = state[robot_domain[0]: robot_domain[2]+1,
        #                   robot_domain[1]: robot_domain[3]+1, 0]
        #     items = list(items[0, :]) + list(items[-1, :])
        #     robot_coor = [robot.get_position[0] - robot_domain[0], robot.get_position[1] - robot_domain[1]]
        #     robot_loc = robot_coor[0]*self.robot_domain_size[0] + robot_coor[1]
        #     writer = csv.writer(file)
        #     row = np.concatenate(([robot_loc], [action], items))
        #     writer.writerow(row)