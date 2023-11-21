#!/home/volta/Documents/ucph/dlgs/venv/bin/python3

import gym
import numpy as np
import time


env = gym.make("CartPole-v1", render_mode='human')

# simulate the environment
episodeNumber=5
timeSteps=8000

"""
initial_state=env.reset()
env.render()
appendedObservations=[]
for timeIndex in range(timeSteps):
    print(timeIndex)
    random_action=env.action_space.sample()
    observation, reward, terminated, truncated, info =env.step(random_action)
    appendedObservations.append(observation)
    time.sleep(0.1)
    if (terminated):
        time.sleep(1)
        break
"""
 
 
for episodeIndex in range(episodeNumber):
    initial_state=env.reset()
    print(episodeIndex)
    env.render()
    appendedObservations=[]
    for timeIndex in range(timeSteps):
        print(timeIndex)
        random_action=env.action_space.sample()
        observation, reward, terminated, truncated, info =env.step(random_action)
        appendedObservations.append(observation)
        time.sleep(0.1)
        if (terminated):
            time.sleep(1)
            break

env.close()
