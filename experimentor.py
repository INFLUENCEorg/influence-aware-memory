import os
import tensorflow as tf
from PPO.PPOcontroller import PPOcontroller
from environments.vectorized_environment import VectorizedEnvironment
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import argparse
import yaml


class Experimentor(object):
    """
    Creates experimentor object to store interact with the environment and
    the agent and log results.
    """

    def __init__(self, parameters):
        """
        Initializes the experiment by extracting the parameters
        @param parameters a dictionary with many obligatory elements
        <ul>
        <li> "env_type" (SUMO, atari, grid_world),
        <li> algorithm (DQN, PPO)
        <li> maximum_time_steps
        <li> maximum_episode_time
        <li> skip_frames
        <li> save_frequency
        <li> step
        <li> episodes
        and more TODO
        </ul>

        @param logger  TODO what is it exactly? It must have the function
        log_scalar(key, stat_mean, self.step[factor_i])
        """
        self.parameters = parameters
        self.path = self.generate_path(self.parameters)
        self.generate_env()
        self.generate_controller(self.env.action_space())
        self.train_frequency = self.parameters["train_frequency"]
        tf.reset_default_graph()

    def generate_path(self, parameters):
        """
        Generate a path to store e.g. logs, models and plots. Check if
        all needed subpaths exist, and if not, create them.
        """
        path = self.parameters['name']
        result_path = os.path.join("results", path)
        model_path = os.path.join("models", path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return path

    def generate_env(self, test_it=0):
        """
        Create environment container that will interact with SUMO
        """
        self.env = VectorizedEnvironment(self.parameters)
        # self.parameters['num_workers'] = self.env.num_workers

    def generate_controller(self, actionmap):
        """
        Create controller that will interact with agents
        """
        if self.parameters['algorithm'] == 'PPO':
            self.controller = PPOcontroller(self.parameters, actionmap)

    def print_results(self, info):
        """
        Prints results to the screen.
        """
        print(("Train step {} of {}".format(self.step,
                                            self.maximum_time_steps)))
        print(("-"*30))
        print(("Episode {} ended after {} steps.".format(
                                    self.controller.episodes,
                                    info['l'])))
        print(("- Total reward: {}".format(info['r'])))
        print(("-"*30))

    def run(self):
        """
        Runs the experiment.
        """
        self.maximum_time_steps = int(self.parameters["max_steps"])
        save_frequency = self.parameters['save_frequency']
        self.step = max(self.parameters["iteration"], 0)
        # reset environment
        step_output = self.env.reset()
        while self.step < self.maximum_time_steps:
            # Select the action to perform
            get_actions_output = self.controller.get_actions(step_output)
            # Increment step
            self.controller.increment_step()
            self.step += 1
            # Get new state and reward given actions a
            next_step_output = self.env.step(get_actions_output['action'],
                                             step_output['obs'])
            if 'episode' in next_step_output['info'][0].keys():
                self.print_results(next_step_output['info'][0]['episode'])
            if self.parameters['mode'] == 'train':
                # Store experiences in buffer.
                self.controller.add_to_memory(step_output, next_step_output,
                                              get_actions_output)
                # Estimate the returns using value function when time
                # horizon has been reached
                self.controller.bootstrap(next_step_output)
                if self.step % self.train_frequency == 0 and \
                   self.controller.full_memory():
                    self.controller.update()
                if self.step % save_frequency == 0:
                    # Tensorflow only stores a limited number of networks.
                    self.controller.save_graph(self.step)
                    self.controller.write_summary()

                step_output = next_step_output

        self.env.close()

def get_config_file():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', default=None, help='config file')
    args = parser.parse_args()
    return args.config

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']

if __name__ == "__main__":
    config_file = get_config_file()
    parameters = read_parameters(config_file)
    exp = Experimentor(parameters)
    exp.run()
