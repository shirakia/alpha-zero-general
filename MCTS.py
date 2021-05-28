import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """
    
    def __init__(self, game, nn_agent, eps=1e-8, cpuct=1.0):
        self.game = game
        self.nn_agent = nn_agent
        self.eps = eps
        self.cpuct = cpuct
        
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Vs = {}  # stores game.getValidMoves for board s
        
        self.last_obs = None

    def getActionProb(self, obs, timelimit=1.0):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """        
        start_time = time.time()
        while time.time() - start_time < timelimit:
            self.search(obs, self.last_obs)

        s = self.game.stringRepresentation(obs)
        i = obs.index
        counts = [
            self.Nsa[(s, i, a)] if (s, i, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]
        prob = counts / np.sum(counts)
        self.last_obs = obs
        return prob

    def search(self, obs, last_obs):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        
        s = self.game.stringRepresentation(obs)

        if s not in self.Ns:
            values = [-10] * 4
            for i in range(4):
                if len(obs.geese[i]) == 0:
                    continue
                    
                # leaf node
                self.Ps[(s, i)], values[i] = self.nn_agent.predict(obs, last_obs, i)
                    
                valids = self.game.getValidMoves(obs, last_obs, i)    
                self.Ps[(s, i)] = self.Ps[(s, i)] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[(s, i)])
                if sum_Ps_s > 0:
                    self.Ps[(s, i)] /= sum_Ps_s  # renormalize

                self.Vs[(s, i)] = valids
                self.Ns[s] = 0
            return values

        best_acts = [None] * 4
        for i in range(4):
            if len(obs.geese[i]) == 0:
                continue
            
            valids = self.Vs[(s, i)]
            cur_best = -float('inf')
            best_act = self.game.actions[-1]

            # pick the action with the highest upper confidence bound
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    if (s, i, a) in self.Qsa:
                        u = self.Qsa[(s, i, a)] + self.cpuct * self.Ps[(s, i)][a] * math.sqrt(
                                self.Ns[s]) / (1 + self.Nsa[(s, i, a)])
                    else:
                        u = self.cpuct * self.Ps[(s, i)][a] * math.sqrt(
                            self.Ns[s] + self.eps)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = self.game.actions[a]
                        
            best_acts[i] = best_act
        
        next_obs = self.game.getNextState(obs, last_obs, best_acts)
        values = self.search(next_obs, obs)

        for i in range(4):
            if len(obs.geese[i]) == 0:
                continue
                
            a = self.game.actions.index(best_acts[i])
            v = values[i]
            if (s, i, a) in self.Qsa:
                self.Qsa[(s, i, a)] = (self.Nsa[(s, i, a)] * self.Qsa[
                    (s, i, a)] + v) / (self.Nsa[(s, i, a)] + 1)
                self.Nsa[(s, i, a)] += 1

            else:
                self.Qsa[(s, i, a)] = v
                self.Nsa[(s, i, a)] = 1

        self.Ns[s] += 1
        return values
