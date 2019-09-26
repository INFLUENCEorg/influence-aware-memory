import gym
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import random
from AbstractEnvironment import AbstractEnvironment
# from sumoai.Factor import Factor
import cv2


class Environment(AbstractEnvironment):
    """
    (PO)MDP enviroment built to interact with the openAI gym.
    """

    def __init__(self, parameters):
        # The parameters of the experiment
        self.parameters = parameters
        # Initialize the openAI environment
        self.env = gym.make(parameters['scene'])
        # if self.parameters['mode'] == 'evaluate':
        #     self.env = gym.wrappers.Monitor(self.env, './videos/'
        #                                     + str(self.parameters['name']))
        # summary_path = os.path.join("results", self.path,
        #                             self.parameters['mode'])
        # self.writer = tf.summary.FileWriter(summary_path)
        # self._create_statistic_collectors()
        # Initialize array to store two most recent frames for max operation
        self.last_frames = np.zeros((2,)+self.env.observation_space.shape,
                                    dtype=np.uint8)
        self.episode_step = 0
        self.lives = 0

    # Override
    def reset(self):
        # Select the new seed for this episode.
        seed = random.choice(list(range(1000)))
        self.env.seed(seed)
        # self._initialize_book_keeping()
        # Observation
        frame = self.env.reset()
        # Take action when reseting for environments that require firing.
        if 'FIRE' in self.env.unwrapped.get_action_meanings():
            frame, _, done, _ = self.env.step(1)
        # Perfom a random number of noop actions at the beginning of an episode
        # to introduce stochasticity
        noop_max = 30
        noop_action = 0
        for k in range(np.random.randint(1, noop_max)):
            frame, _, done, _ = self.env.step(noop_action)
            if done:
                frame = self.env.reset()
        # self._visualize_image(frame)
        processed_frame = self._preprocess(frame, self.parameters['frame_height'],
                                           self.parameters['frame_width'],
                                           self.parameters['box_center'],
                                           self.parameters['box_height'],
                                           self.parameters['box_width'])
        s = np.array([processed_frame]*self.parameters['num_frames'])
        s = np.swapaxes(s, 0, 2)
        s = np.swapaxes(s, 0, 1)
        self.s = s
        return s, -1

    # Override
    def step(self, action):
        """
        """
        if self.episode_step == 0:
            print(action)
        self.episode_step += 1
        if self.parameters['gui'] and self.parameters['num_workers'] == 1:
            self.env.render()
        total_reward = 0.0

        for i in range(self.parameters['skip_frames']):
            lives = self.env.unwrapped.ale.lives()
            # Take action when loosing lives for environments that require firing.
            if 'FIRE' in self.env.unwrapped.get_action_meanings() and lives < self.lives:
                new_frame, reward, done, info = self.env.step(1)
            else:
                new_frame, reward, done, info = self.env.step(action)
            if i == self.parameters['skip_frames'] - 2:
                self.last_frames[0] = new_frame
            if i == self.parameters['skip_frames'] - 1:
                self.last_frames[1] = new_frame
            total_reward += reward
            self.lives = lives
            if done:
                break
        # statement below resets the environment if this happens
        if done:
            self.reset()
        # Note that the observation on the done=True frame doesn't matter
        # Max operation on the two most recent frames to prevent flickering
        max_frame = self.last_frames.max(axis=0)
        # Add flickering to make the environment partially observable
        if self.parameters['flicker']:
            p = 0.5
            prob_flicker = random.uniform(0, 1)
            if prob_flicker > p:
                max_frame = np.zeros_like(max_frame)
        # max_frame, total_reward, done, info = self.env.step(actions[0])
        new_s = np.zeros((self.parameters['frame_height'],
                          self.parameters['frame_width'],
                          self.parameters['num_frames']))
        prev_s = np.copy(self.s)
        processed_frame = self._preprocess(max_frame,
                                           self.parameters['frame_height'],
                                           self.parameters['frame_width'],
                                           self.parameters['box_center'],
                                           self.parameters['box_height'],
                                           self.parameters['box_width'])
        new_s[:, :, 0] = processed_frame
        new_s[:, :, 1:] = prev_s[:, :, :-1]
        # with self.graph.as_default():
        #     key = tf.get_default_graph().get_tensor_by_name('episodes/rewards:0')
        #     self.feed_dict[key].append(total_reward)
        self.s = new_s
        return new_s, total_reward, done, action, new_frame

    # Override
    def close(self, episode, full_memory=True):
        if self.parameters['gui']:
            self.env.render()
        # Log the environment information. Writes results to tensorflow summaries.
        if full_memory:
            self.log(episode)

    # Override
    def action_space(self):
        return list(range(self.env.action_space.n))

    # Override
    def getActions(self) -> list:
        return list(range(self.env.action_space.n))

    """******** PRIVATE FUNCTIONS *******"""

    def _preprocess(self, new_frame, frame_height, frame_width,
                    box_center=None, box_height=None, box_width=None):

        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(new_frame, (frame_height, frame_width),
                         interpolation=cv2.INTER_AREA)
        return obs

    def _visualize_image(self, frame):
        from PIL import ImageEnhance
        img = Image.fromarray(frame)
        w, h = img.size
        print(w, h)
        blurred = img.crop((0, 0, w, 145))
        enhancer = ImageEnhance.Brightness(blurred)
        # blurred = np.asarray(blurred.filter(ImageEnhance.BLUR))
        blurred = enhancer.enhance(1.0)
        not_blurred = np.asarray(img.crop((0, 145, w, h)))
        img = np.concatenate((blurred, not_blurred))
        Image.fromarray(img).show()
        input("")
