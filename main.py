from collections import defaultdict
import pickle

import gymnasium as gym
import numpy as np

env = gym.make('Taxi-v3', render_mode="human")
env.reset()

# Hyperparameters
LEARNING_RATE = 0.11
DISCOUNT_FACTOR = 0.99

EXPLORATION_RATE = 1.0
EXPLORATION_RATE_DECAY = 0.20002
EXPLORATION_RATE_MIN = 0.01

def get_action_space_n_zeros():
    return np.zeros(env.action_space.n)

class CarAgent():
    def __init__(self, env: gym.Env, learning_rate: float, 
                 initial_epsilon: float, epsilon_decay: float, 
                 final_epsilon: float, discount_factor: float):

        self.env = env

        self.q_values = defaultdict(get_action_space_n_zeros)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, observation: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[observation]))
        
    def update(self, observation: tuple[int, int, bool], action: int, 
               reward: float, terminated: bool, 
               next_observation: tuple[int, int, bool]):
        future_q_value = (not terminated) * np.max(self.q_values[next_observation])

        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.q_values[observation][action]

        self.q_values[observation][action] = (
            self.q_values[observation][action] + self.learning_rate * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def serialize(self):
        with open("q_values.bin", "wb") as file:
            pickle.dump(self.q_values, file)

    def unserialize(self):
        with open("q_values.bin", "rb") as file:
            self.q_values = pickle.load(file)

agent = CarAgent(env, LEARNING_RATE, EXPLORATION_RATE, EXPLORATION_RATE_DECAY, 
                 EXPLORATION_RATE_MIN, DISCOUNT_FACTOR)

agent.unserialize()

for episode in range(30):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)

        agent.update(observation, action, reward, terminated, next_observation)

        done = terminated or truncated
        observation = next_observation

    agent.decay_epsilon()

agent.serialize()

env.close()

