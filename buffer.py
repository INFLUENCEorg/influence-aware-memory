import numpy as np
import random
import pickle


class Buffer(dict):
    """
    Dictionary to store transitions in.
    """
    # NOTE: The replay memory is modified so that transitions are stored in a
    # dictionary.

    def __init__(self, parameters, act_size, separate=False):
        """
        Initialize the memory with the right size.
        """

        self.memory_size = parameters['memory_size']
        self.batch_size = parameters['batch_size']
        self.height = parameters['frame_height']
        self.width = parameters['frame_width']
        self.frames = parameters['num_frames']
        self.act_size = act_size
        if parameters['influence']:
            self.seq_len = parameters['inf_seq_len']
        elif parameters['recurrent']:
            self.seq_len = parameters['seq_len']
        else:
            self.seq_len = 1

    def __getitem__(self, key):
        if key not in self.keys():
            self[key] = list()
        return super(Buffer, self).__getitem__(key)

    def sample(self, batch_size):
        """
        Sample a batch from the dataset. This can be implemented in
        different ways.
        """
        raise NotImplementedError

    def get_latest_entry(self):
        """
        Retrieve the entry that has been added last. This can differ
        between sequence en non-sequence samplers.
        """
        raise NotImplementedError

    def full(self):
        """
        Check whether the replay memory has been filled.
        """
        # TODO: returns are only calculated either when time horizon is reached
        # or when the episode is over. When that happens all fields in replay_
        # memory
        # are the same size
        # This means replay memory could
        if 'returns' not in self.keys():
            return False
        else:
            return len(self['returns']) >= self.memory_size

    def store(self, path):
        """
        Store the necessary information to recreate the replay memory.
        """
        with open(path + 'buffer.pkl', 'wb') as f:
            pickle.dump(self.buffer, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """
        Load a stored replay memory.
        """
        with open(path + 'replay_memory.pkl', 'rb') as f:
            self.buffer = pickle.load(f)

    def empty(self):
        for key in self.keys():
            self[key] = []


class SerialSampling(Buffer):
    """
    Batches of experiences are sampled in a series one after the other.
    This ensures all experiences are used the same number of times. Valid for
    sequences or single experiences.
    """

    def __init__(self, parameters, act_size):
        """
        This is an instance of a plain Replay Memory object. It does not
        need more information than its super class.
        """
        super().__init__(parameters, act_size)

    def sample(self, b, n_sequences, keys=None):
        """
        """
        batch = {}
        if keys is None:
            keys = self.keys()
        for key in keys:
            batch[key] = []
            for s in range(n_sequences):
                start = s*self.seq_len + b*self.batch_size
                end = (s+1)*self.seq_len + b*self.batch_size
                batch[key].extend(self[key][start:end])
            # permut dimensions workers-batch to mantain sequence order
            axis = np.arange(np.array(batch[key]).ndim)
            axis[0], axis[1] = axis[1], axis[0]
            batch[key] = np.transpose(batch[key], axis)
        return batch

    def shuffle(self):
        """
        """
        n = len(self['returns'])
        # Only include complete sequences
        indices = np.arange(0, n - n % self.seq_len, self.seq_len)
        random.shuffle(indices)
        for key in self.keys():
            shuffled_memory = []
            for i in indices:
                shuffled_memory.extend(self[key][i:i+self.seq_len])
            self[key] = shuffled_memory

    def get_last_entries(self, t, keys=None):
        """
        """
        if keys is None:
            keys = self.keys()
        batch = {}
        for key in keys:
            batch[key] = self[key][-t:]
        return batch

    def zero_padding(self, missing, worker):
        for key in self.keys():
            if key not in ['advantages', 'returns']:
                padding = np.zeros_like(self[key][-1])
                for i in range(missing):
                    self[key].append(padding)


class RandomSampling(Buffer):
    """
    Database to sample random batches from. (does not sample sequences)
    """

    def __init__(self, parameters, act_size):
        """
        This is an instance of a plain Replay Memory object. It does not
        need more information than its super class.
        """
        super().__init__(parameters, act_size)

    def sample(self, keys=None):
        """
        Sample a batch from the dataset by chosing the transitions at
        random.
        """
        n = len(self['obs'])
        indices = np.random.randint(n, size=self.batch_size)

        batch = {}
        if keys is None:
            keys = self.keys()

        for key in keys:
            batch[key] = [self[key][i] for i in indices]

        return batch

    def get_last_entries(self, t, keys=None):
        """
        """
        if keys is None:
            keys = self.keys()
        batch = {}
        for key in keys:
            batch[key] = self[key][-t:]

        return batch


class RandomSequenceSampling(Buffer):
    """
    Database to sample random batches from. This class samples sequences
    of the same size as there are frames.
    """

    def __init__(self, parameters,act_size):
        """
        This is an instance of a plain Replay Memory object. A sequence
        is of length frames and the number of frames in the states is 1.
        """
        super().__init__(parameters, act_size)

    def sample(self, keys=None):
        """
        Sample a batch from the database by chosing random starting points
        in the dataset and then appending a sequence to the batch.
        Returns:
            a batch of size batch_size x seq_len
        """
        n = len(self['obs'])
        n = n-self.seq_len+1

        batch = {}
        if keys is None:
            keys = self.keys()

        for key in keys:
            batch[key] = []

        for b in range(self.batch_size):
            start=np.random.choice(np.arange(n))
            while np.sum(self['dones'][start:start+self.seq_len])!=0:
                start = np.random.choice(np.arange(n))
            end = start + self.seq_len
            for key in keys:
                batch[key]+=self[key][start:end]

        return batch


    def get_sequence(self, index, collection):
        """
        Retrieves the frames that come before the item at index.
        But we want the item at index to be included.
        Example:
            If we start at index 4 and the frame size is 4
            we want the sequence to be: [1, 2, 3, 4]
            NOT [0, 1, 2, 3]
            NOT [4, 3, 2, 1] or [3, 2, 1, 0]
        """
        # Because the index operation does not include the upper bound
        stop = index + 1
        start = stop - self.seq_len

        if start < 0 and stop >= 0:
            try:
                seq = np.vstack((collection[start:], collection[:stop]))
            except ValueError:
                seq = np.append(collection[start:], collection[:stop])
        else:
            seq = collection[start:stop]

        # The append operation adds an extra dimension to the matrix
        if len(seq.shape) != len(collection.shape):
            seq = np.reshape(seq, (-1,))

        return seq

    def get_latest_entry(self):
        """
        Retrieve the latest sequence that can be retrieved from the
        replay memory.
        """
        selected_states = self.get_sequence(self.pointers-1, self.states)
        selected_next_states = self.get_sequence(self.pointers-1, self.next_states)

        if self.full_seq:
            selected_actions = self.get_sequence(self.pointers-1, self.actions)
            selected_rewards = self.get_sequence(self.pointers-1, self.rewards)
        else:
            selected_actions = [self.actions[self.pointers - 1]]
            selected_rewards = [self.rewards[self.pointers-1]]

        return selected_states, selected_actions, selected_rewards, selected_next_states


class ActionObservationSampling(RandomSequenceSampling):
    """
    Database to sample random batches from. This class samples sequences
    of the same size as there are frames but also returns the actions that happened
    to get to the state..
    """

    def __init__(self, memory_size, height, width, frames, batch_size, full_seq=False):
        RandomSequenceSampling.__init__(self, memory_size, height, width, frames, batch_size, full_seq)

    def append(self, state, action, reward, next_state, terminal):
        """
        Add the given information to the Replay Memory.
        """
        self.states[self.pointers] = state[0]
        self.actions[self.pointers] = action
        self.rewards[self.pointers] = reward
        self.next_states[self.pointers] = next_state[0]
        self.terminal_states[self.pointers] = int(terminal)

        self.pointers += 1
        self.items += 1
        if self.pointers == self.memory_size:
            # Restart from 0
            self.pointers = 0

    def sample(self):
        """
        Sample a batch from the database by chosing random starting points
        in the dataset and then appending a sequence to the batch.
        Returns:
            a batch of size batch_size x seq_len
        """
        batch_length = self.batch_size*self.seq_len

        selected_states = np.zeros((batch_length, self.height, self.width, 1))
        selected_next_states = np.zeros((batch_length, self.height, self.width, 1))
        selected_actions = np.zeros((batch_length,))
        selected_rewards = np.zeros((batch_length,)) if self.full_seq else np.array([])
        selected_prev_actions = np.zeros((batch_length,))

        indices = np.arange(self.seq_len-1, self.memory_size)

        for b in np.arange(0, batch_length, self.seq_len):
            # Select a sequence of states that does not contain a terminal state
            # except for at the end.
            i = random.choice(indices)
            while (sum(self.terminal_states[i+1-self.seq_len:i+1]) > 0 and self.terminal_states[i] != 1):
                i = random.choice(indices)
            # Append the sequence information to the batch
            selected_states[b:b+self.seq_len] = self.get_sequence(i, self.states)
            selected_actions[b:b+self.seq_len] = self.get_sequence(i, self.actions)

            if self.full_seq:
                selected_rewards[b:b+self.seq_len] = self.get_sequence(i, self.rewards)
            else:
                selected_rewards = np.append(selected_rewards, self.rewards[i])

            selected_next_states[b:b+self.seq_len] = self.get_sequence(i, self.next_states)
            selected_prev_actions[b:b+self.seq_len] = self.get_sequence(i-1, self.actions)

        return selected_states, np.expand_dims(selected_actions,1), selected_rewards, selected_next_states, np.expand_dims(selected_prev_actions,1)

    def get_latest_entry(self):
        """
        Retrieve the latest sequence that can be retrieved from the
        replay memory.
        """
        selected_states = self.get_sequence(self.pointers-1, self.states)
        selected_next_states = self.get_sequence(self.pointers-1, self.next_states)
        selected_actions = self.get_sequence(self.pointers-1, self.actions)
        selected_prev_actions = self.get_sequence(self.pointers-2, self.actions)

        if self.full_seq:
            selected_rewards = self.get_sequence(self.pointers-1, self.rewards)
        else:
            selected_rewards = [self.rewards[self.pointers-1]]

        return selected_states, np.expand_dims(selected_actions,1), selected_rewards, selected_next_states, np.expand_dims(selected_prev_actions,1)
