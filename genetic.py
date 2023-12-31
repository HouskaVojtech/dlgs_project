import numpy as np
import gym


def single_relu(x):

    if x < 0:
        return 0
    else:
        return x


relu = np.vectorize(single_relu)


class Agent:

    def __init__(self, reservoir_size, in_size, out_size, std=0.5, verbose=False):

        self.verbose = verbose
        self.std = std

        self.reward = 0

        self.in_weights = np.random.normal(0, std, [reserovir_size, in_size])
        self.reservoir = np.random.normal(0, std, [reserovir_size, reservoir_size])

        self.bias1 = .1 * np.ones(4)

        self.weights2 = np.random.normal(0, std, [2, 4])
        self.bias2 = .1 * np.ones(2)


    def print(self, text=''):
        if self.verbose:
            print(text)

    def mutate(self, std=None):

        if not std:
            std = self.std

        weights1 = self.weights1 + np.random.normal(0, std, [4, 4])
        weights2 = self.weights2 + np.random.normal(0, std, [2, 4])
        bias1 = self.bias1 + np.random.normal(0, std, 4)
        bias2 = self.bias2 + np.random.normal(0, std, 2)

        return weights1, bias1, weights2, bias2

    def offspring(self, std=None):

        if not std:
            std = self.std

        weights1, bias1, weights2, bias2 = self.mutate(std)

        agent = Agent(std)

        agent.weights1 = weights1
        agent.weights2 = weights2
        agent.bias1 = bias1
        agent.bias2 = bias2

        return agent

    def action(self, observation):
        """
        The agent's "brain" is consists of a two-layer neural network, which is fed the environment's observation.
        :param observation: the environment's observation output.
        :return: the action that has the highest score.
        """

        result = np.dot(self.weights1, observation)
        self.print(result)

        result += self.bias1
        self.print(result)

        result = relu(result)
        self.print(result)

        result = np.dot(self.weights2, result)
        self.print(result)

        result += self.bias2
        self.print(result)

        return np.argmax(result)

    def add_reward(self, reward):
        self.reward += reward

    def reset(self):
        self.reward = 0


class Generation:

    def __init__(self, n=10, std=.1, env=None, verbose=False):

        self.generation = 0
        self.std = std

        self.n = n
        self.agents = [Agent(std=std, verbose=verbose) for _ in range(n)]

        if env is None:
            self.env = gym.make('CartPole-v0')
        else:
            self.env = env

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def select(self, proportion=.3, std=None):
        p = np.array([agent.reward for agent in self.agents])
        p /= p.sum()

        agents = np.random.choice(self.agents, size=int(proportion*self.n), replace=False, p=p)

        p = np.array([agent.reward for agent in agents])
        p /= p.sum()

        n = self.n - int(proportion*self.n)

        if std is None:
            std = self.std

        self.agents = [agent for agent in agents] + \
                      [agent.offspring(std) for agent in np.random.choice(agents, n, p=p)]

    def simulation_step(self, n=100):

        for agent in self.agents:

            observation = self.env.reset()

            for t in range(n):
                observation, reward, done, info = self.env.step(agent.action(observation))

                agent.add_reward(reward)

                if done:
                    break

        self.generation += 1

        return np.array([agent.reward for agent in self.agents])

    def simulation(self, n):

        for i in range(n):
            self.reset()
            self.simulation_step()
            self.select()
