import multiprocessing
import multiprocessing.connection
from environments import atari_environment
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines import bench
import os


def worker_process(remote: multiprocessing.connection.Connection, parameters, worker_id):
    # if parameters['env_type'] == 'SUMO':
        # env = SUMO_environment.Environment(parameters)
    # elif parameters['env_type'] == 'atari':
    # env = atari_environment.Environment(parameters)
    log_dir = './log'
    env = make_atari(parameters['scene'])
    env = bench.Monitor(
                env,
                os.path.join(log_dir, str(worker_id)),
                allow_early_resets=False)
    env = wrap_deepmind(env)
    # elif parameters['env_type'] == 'gridworld':
        # env = grid_world_environment.Environment(parameters)

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            if done is True:
                obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'action_space':
            remote.send(env.action_space.n)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker(object):
    # child: multiprocessing.connection.Connection
    # process: multiprocessing.Process

    def __init__(self, parameters, worker_id):

        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process,
                                               args=(parent, parameters, worker_id))
        self.process.start()
