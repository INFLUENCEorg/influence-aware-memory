import os
import tensorflow as tf
from PPO.PPOcontroller import PPOcontroller
# from DQN.DQNcontroller import DQNcontroller
from environments.vectorized_environment import VectorizedEnvironment
from sacred import Experiment
from sacred.observers import FileStorageObserver
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


class Experimentor(object):
    """
    Experimentor class built to store experiment parameters, log results and
    interact with the environment and the agent(s).
    """

    def __init__(self, parameters: dict, logger):
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
        self.logger = logger
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
        if self.parameters['algorithm'] == 'DQN':
            self.controller = DQNcontroller(self.parameters, actionmap, self.logger)
        elif self.parameters['algorithm'] == 'PPO':
            self.controller = PPOcontroller(self.parameters, actionmap, self.logger)

    def print_results(self, info):
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
        Runs the experiment by looping over time.
        """
        self.maximum_time_steps = int(self.parameters["max_steps"])
        save_frequency = self.parameters['save_frequency']
        self.step = max(self.parameters["iteration"], 0)
        """
        Train the model
        """
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
                # Store the transition in the replay memory.
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
                # if step_output['done']:
                #     break
        self.env.close()


ex = Experiment()
# ex.add_config('./config.yaml') # default path

@ex.config
def add_slurm_id():
    # If we run the experiment on the cluster, add the slurm id
    if 'SLURM_JOB_ID' in os.environ:
        slurm_id = os.environ['SLURM_JOB_ID']

@ex.automain
def my_main(parameters, _run):
    # Check if experiment was initialized correctly
    # exp_file_storage = ex.current_run.meta_info['options']['--file_storage'] + '/' + str(ex.current_run._id) + '/'
    # # If there is no run.json
    # if not os.path.isfile(exp_file_storage + 'run.json'):
        # raise Exception("Experiment failed to initialize correctly, run.json was not created")
    # If the run.json is empty
    # elif os.stat(exp_file_storage + 'run.json').st_size == 0:
    #     raise Exception("Experiment failed to initialize correctly, run.json is empty")

    exp = Experimentor(parameters, _run)
    exp.run()
