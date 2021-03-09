import gym
import random
import numpy
import time

env = gym.make("Taxi-v3")

alpha = 0.9
gamma = 0.9
num_of_episodes = 1000
num_of_steps = 500
epsilon = 0.2

Q_reward = numpy.zeros((500,6))

def train():
    for episode in range(num_of_episodes):
        state = env.reset()

        for step in range(num_of_steps):
            action = 0
            if numpy.random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = numpy.argmax(Q_reward[state])

            q_current = Q_reward[state][action]
            next_state, reward, done, _ = env.step(action)
            max_q_next = numpy.max(Q_reward[next_state])

            Q_reward[state][action] = q_current + alpha * (reward + gamma * max_q_next - q_current)

            state = next_state
            if done:
                break
        print(f"Episode {episode} completed")

train()

total_rewards = []
total_steps = []
def runTaxi(test_num):
    state = env.reset()
    tot_reward = 0
    for t in range(50):
        action = numpy.argmax(Q_reward[state])
        state, reward, done, _ = env.step(action)
        tot_reward += reward
        if done:
            print(f"Total reward of test number {test_num} is {tot_reward}")
            total_steps.append(t + 1)
            break

    total_rewards.append(tot_reward)
    
test_episodes = 10
for i in range(test_episodes):
    runTaxi(i + 1)

#Best result : 8.9 points in avg and 12.1 steps in avg
avg_rewards = numpy.mean(total_rewards)
avg_steps = numpy.mean(total_steps)
print(f"Average total rewards: {avg_rewards}")
print(f"Average total steps: {avg_steps}")