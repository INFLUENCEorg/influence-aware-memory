from environments.warehouse2.item import Item
from environments.warehouse2.robot import Robot
from environments.warehouse2.utils import *
import numpy as np
from gym import spaces
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MiniWarehouse(gym.Env):
    """
    warehouse environment
    """

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}
            #    4: 'NOOP'}

    OBS_SIZE = 37

    def __init__(self, seed):
        self.n_columns = 5
        self.n_rows = 5
        self.n_robots_row = 1
        self.n_robots_column = 1
        self.distance_between_shelves = 4
        self.robot_domain_size = [5, 5]
        self.prob_item_appears = 0.04
        # The learning robot
        self.learning_robot_id = 0
        self.max_episode_length = 100
        self.render_bool = False
        self.render_delay = 0.5
        self.obs_type = 'vector'
        self.items = []
        self.img = None
        # self.reset()
        self.max_waiting_time = 8
        self.total_steps = 0
        self.reset()
        self.seed(seed)

    ############################## Override ###############################

    def reset(self):
        """
        Resets the environment's state
        """
        self.robot_id = 0
        self._place_robots()
        self.item_id = 0
        self.items = []
        self._remove_items()
        self._add_items()
        obs = self._get_observation()
        self.episode_length = 0
        self.prev_items = np.copy(self.items)
        # self.max_waiting_time = np.random.choice([4, 8], 4) 
        return obs

    def step(self, action):
        """
        Performs a single step in the environment.
        """
        self._robots_act([action])
        self._increase_item_waiting_time()
        dset = self.get_dset
        reward = self._compute_reward(self.robots[self.learning_robot_id])
        self._remove_items()
        infs = self.get_infs
        self._add_items()
        obs = self._get_observation()
        self.prev_items = np.copy(self.items)
        # Check whether learning robot is done
        # done = self.robots[self.learning_robot_id].done
        self.total_steps += 1
        self.episode_length += 1
        done = (self.max_episode_length <= self.episode_length)
        # if self.render_bool:
        #     self.render(self.render_delay)
        return obs, reward, done, {'dset': dset, 'infs': infs}

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(self.OBS_SIZE,))

    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        return spaces.Discrete(len(self.ACTIONS))
    
    @property
    def get_dset(self):
        state = self._get_state()
        robot = self.robots[self.learning_robot_id]
        obs = robot.observe(state, 'vector')
        # dset = obs[49:]
        dset = obs#[25:]
        return dset

    @property
    def get_infs(self):
        state_bitmap = np.zeros([self.n_rows, self.n_columns], dtype=np.int)
        for item in self.prev_items:
            if item.get_id not in [item.get_id for item in self.items] and  item.get_id not in self.removed_by_robot:
                item_pos = item.get_position
                state_bitmap[item_pos[0], item_pos[1]] = 1
        infs = np.concatenate((state_bitmap[0, 1:-1].flatten(),
                               state_bitmap[1:-1, [0,-1]].flatten(),
                               state_bitmap[-1, 1:-1].flatten())
        )
        return infs


    def render(self, mode='human'):
        """
        Renders the environment
        """
        bitmap = self._get_state()
        position = self.robots[self.learning_robot_id].get_position
        bitmap[position[0], position[1], 1] += 1
        for robot_id, robot in enumerate(self.robots):
            # if robot.is_slow:
            position = robot.get_position
            bitmap[position[0], position[1], 1] += 2
        im = bitmap[:, :, 0] - 2*bitmap[:, :, 1]

        if self.img is None:
            fig,ax = plt.subplots(1)
            self.img = ax.imshow(im, vmin=-3, vmax=1)
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
                self.img.axes.get_xaxis().set_visible(False)
                self.img.axes.get_yaxis().set_visible(False)
        else:
            self.img.set_data(im)
        # plt.pause(delay)
        plt.savefig('images/image.jpg')
        img = plt.imread('images/image.jpg')
        return img

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

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
                                                  robot_domain, False))
                self.robot_id += 1

    def _add_items(self):
        """
        Add new items to the designated locations in the environment.
        """
        item_locs = None
        if len(self.items) > 0:
            item_locs = [item.get_position for item in self.items]
        for row in range(self.n_rows):
            if row % (self.distance_between_shelves) == 0:
                for column in range(1, self.n_columns):
                    if column % (self.distance_between_shelves) != 0:
                        loc = [row, column]
                        loc_free = True
                        region_free = True
                        just_removed = loc not in self.just_removed_list
                        if item_locs is not None:
                            loc_free = loc not in item_locs
                        if np.random.uniform() < self.prob_item_appears and loc_free and just_removed:
                            self.items.append(Item(self.item_id, loc))
                            self.item_id += 1
                            item_locs = [item.get_position for item in self.items]
            else:
                for column in range(0, self.n_rows, self.distance_between_shelves):
                    loc = [row, column]
                    loc_free = True
                    just_removed = loc not in self.just_removed_list
                    if item_locs is not None:
                        loc_free = loc not in item_locs and loc 
                    if np.random.uniform() < self.prob_item_appears and loc_free and just_removed:
                        self.items.append(Item(self.item_id, loc))
                        self.item_id += 1
                        item_locs = [item.get_position for item in self.items]



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
        item_waiting_times = [item.get_waiting_time for item in self.items]
        sorted_indices = np.argsort(item_waiting_times)[::-1]
        # if len(sorted_indices) > 0:
        #     print(item_waiting_times)
        #     print(sorted_indices[0])
        for index, item in enumerate(self.items):
            item_pos = item.get_position
            # reward -= item.get_waiting_time*0.1
            if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                # if item.get_waiting_time == 8:
                # (self.initial-self.final)*(1 - step/self.total_steps) + self.final
                # reward += 1 - (item.get_waiting_time - 1)/99
                # if index == 0:
                # if index == sorted_indices[0]:
                # if item_waiting_times[index] == item_waiting_times[0]:
                #     reward = 1
                #     self.items.remove(item)
                #     break
                # else:
                #     reward = 0.1
                #     self.items.remove(item)
                #     break
                reward = item_waiting_times[index]/max(item_waiting_times)
                # self.items.remove(item)
                # reward += 10
                # reward += 1/item.get_waiting_time
        return reward


    def _remove_items(self):
        """
        Removes items collected by robots. Robots collect items by steping on
        them
        """
        self.removed_by_robot = []
        self.just_removed_list = []
        for robot in self.robots:
            robot_pos = robot.get_position
            for item in np.copy(self.items):
                item_pos = item.get_position
                if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                    self.items.remove(item)
                    self.removed_by_robot.append(item.get_id)
                    self.just_removed_list.append(item.get_position)
        for item in np.copy(self.items):
            if item.get_waiting_time >= self.max_waiting_time:
                    self.items.remove(item)
                    self.just_removed_list.append(item.get_position)
            # elif item_pos[0] == 0 and item.get_waiting_time >= self.max_waiting_time[0]:
            #     self.items.remove(item)
            # elif item_pos[0] == 4 and item.get_waiting_time >= self.max_waiting_time[1]:
            #     self.items.remove(item)
            # elif item_pos[1] == 0 and item.get_waiting_time >= self.max_waiting_time[2]:
            #     self.items.remove(item)
            # elif item_pos[1] == 4 and item.get_waiting_time >= self.max_waiting_time[3]:
            #     self.items.remove(item)

    def _increase_item_waiting_time(self):
        """
        Increases items waiting time
        """
        for item in self.items:
            item.increase_waiting_time()
