# from sumoai.Factor import Factor

class AbstractEnvironment:
    """
    Abstract class.
    Specifies the functions to be implemented by our environments.
    """

    def __init__(self, path, parameters:dict, test_it=0):
        """
        @param path a path to a directory where we can log results
        @param parameters a dictionary with all kinds of options TODO document in detail. Are these generic?
        @param test_it an optional number to append to log files.
        """
        pass

    def start(self, episode:int) -> dict:
        """
        Resets the environment to the very first (initial) state, s_0, and
        returns this state
        @param episode the episode number
        @return dictionary typically {'obs': [s], 'prev_action': [[-1]]} where s
        is some numpy array.
        """
        pass


    def step(self, actions, state) -> dict:
        """
        Takes the action in the environment, runs a simulation step and then
        computes the new state and reward, and returns these.
        @param actions a list of action-indices for each factor,
        where factors are in the same order as in the factorgrahp.
        getActions(factor)[action-index] then is the action to be taken
        @param state ???
        @return reward, a dict like {'obs': new_s, 'reward': [reward], 'done': [done], 'prev_action': actions}
        """
        pass


    def close(self, episode:int, full_memory=True):
        """
        Closes the environment.
        @param episode the episode number
        @param full_memory if true, Log the environment information. Writes results to tensorflow summaries
        """
        pass

    def getActionMap(self) -> dict:
        """
        @return An action map dictionary.
        Key is the factor number N (index of the factor in the factorgraph, starting at 0).
        value getActions(factor N).

        Example return value for SUMO: {0: [('rrGG',), ('GGrr',)]}
        0 is the factor number, and since each factor has 1 light only, all the actions
        are only for that single traffic light.
        """
        raise Exception("Not implemented")

    def getActions(self) -> list:
        """
        @return a list of allowed actions for the given factor.
                each value is a list of length M where M is the number of agents (traffic lights)
        in factor N. Each element is the allowed actions
        Example return value for SUMO: [('rrGG',), ('GGrr',)]
        """
        raise Exception("Not implemented")
