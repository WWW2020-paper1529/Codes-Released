"""
Implements the gridworld MG.

"""

import numpy as np
import numpy.random as rn

class Gridworld(object):
    """
    Gridworld MG.
    """

    def __init__(self, play_num, grid_size, discount):
        """
        grid_size: Grid size. int.
        play_num: players number
        discount: MG discount. float.
        -> Gridworld
        """

        self.actions = (((1, 0),(1, 0)),((1, 0),(0, 1)),((1, 0),(-1, 0)),((1, 0),(0, -1)),((0, 1),(1, 0)),((0, 1),(0, 1)),((0, 1),(-1, 0)),((0, 1),(0, -1)),((-1, 0),(1, 0)),((-1, 0),(0, 1)),((-1, 0),(-1, 0)),((-1, 0),(0, -1)),((0, -1),(1, 0)),((0, -1),(0, 1)),((0, -1),(-1, 0)),((0, -1),(0, -1)))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**4
        self.grid_size = grid_size
        self.discount = discount
        self.n_players = play_num
        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding joint coordinate.

        i: State int.
        -> ((x1, y1),(x2,y2)) tuple of int tuple.
        """
        coor1 = i // (self.grid_size**2)
        coor2 = i % (self.grid_size**2)

        return ((coor1 % self.grid_size, coor1 // self.grid_size),(coor2 % self.grid_size, coor2 // self.grid_size))

    def point_to_int(self, p):
        """
        Convert a joint coordinate into the corresponding state int.

        p: ((x1, y1),(x2,y2)) tuple of int tuple.
        -> State int.
        """

        return (p[0][1] + p[0][0]*self.grid_size)*(self.grid_size**2) + (p[1][1] + p[1][0]*self.grid_size)

    def neighbouring(self, i, k):
        """
        Get whether two state neighbour each other. Also returns true if they
        are the same point.

        i: ((x1, y1),(x2, y2)) tuple of int tuple.
        k: ((x1, y1),(x2, y2)) tuple of int tuple.
        -> bool.
        """

        return abs(i[0][0] - k[0][0]) + abs(i[0][1] - k[0][1]) <= 1 and abs(i[1][0] - k[1][0]) + abs(i[1][1] - k[1][1]) <= 1

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        ((xi1, yi1),(xi2, yi2)) = self.int_to_point(i)
        ((xj1, yj1),(xj2, yj2)) = self.actions[j]
        ((xk1, yk1),(xk2, yk2)) = self.int_to_point(k)

        # Is original state the destination?
        if (xi1, yi1) == (0,1) or (xi2, yi2) == (0,1):
            return 0

        # Is k the intended state to move to?
        if (xi1 + xj1, yi1 + yj1, xi2 + xj2, yi2 + yj2) == (xk1, yk1,xk2, yk2):
            return 1
        return 0


    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        ((xi1, yi1),(xi2, yi2)) = self.int_to_point(state_int)
        rew = (0,0)

        if xi1 == xi2 and yi1 == yi2:
            rew = (rew[0]-10,rew[1]-10)
        
        if xi1 == 0 and yi1 == 1:
            rew = (rew[0]+5,rew[1])

        if xi2 == 0 and yi2 == 1:
            rew = (rew[0],rew[1]+5)

        return rew

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                                    random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)

    