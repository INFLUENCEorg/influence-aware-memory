from worker import Worker
import multiprocessing as mp
import numpy as np

class VectorizedEnvironment(object):

    def __init__(self, parameters):
        if parameters['num_workers'] < mp.cpu_count():
            self.num_workers = parameters['num_workers']
        else:
            self.num_workers = mp.cpu_count()
        print("*******************************")
        print("The number of workers is: " + str(self.num_workers))
        print("*******************************")
        self.workers = [Worker(parameters, i) for i in range(self.num_workers)]
        self.parameters = parameters

    def reset(self):
        for worker in self.workers:
            worker.child.send(('reset', None))
        output = {'obs': [], 'prev_action': []}
        for worker in self.workers:
            obs = worker.child.recv()
            stacked_obs = np.zeros((self.parameters['frame_height'],
                                    self.parameters['frame_width'],
                                    self.parameters['num_frames']))
            stacked_obs[:, :, 0] = obs[:, :, 0]
            output['obs'].append(stacked_obs)
            output['prev_action'].append(-1)
        return output

    def step(self, actions, stacked_obs):
        for worker, action in zip(self.workers, actions):
            worker.child.send(('step', action))
        output = {'obs': [], 'reward': [], 'done': [], 'prev_action': [],
                  'info': []}
        i = 0
        for worker in self.workers:
            obs, reward, done, info = worker.child.recv()
            stacked_obs[i][:, :, 1:] = stacked_obs[i][:, :, :-1]
            stacked_obs[i][:, :, 0] = obs[:, :, 0]
            output['obs'].append(stacked_obs[i])
            output['reward'].append(reward)
            output['done'].append(done)
            output['info'].append(info)
            i += 1
        output['prev_action'] = actions
        return output

    def action_space(self):
        self.workers[0].child.send(('action_space', None))
        action_space = self.workers[0].child.recv()
        return action_space

    def close(self):
        for worker in self.workers:
            worker.child.send(('close', None))
