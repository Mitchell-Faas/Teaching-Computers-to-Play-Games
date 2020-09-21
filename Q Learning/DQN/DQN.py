import gym
import numpy as np
import matplotlib.pyplot as plt

# Set up the taxi environment
env = gym.make('Taxi-v3')

# Define the discount rate
gamma = 0.9

# Set up the Q table
rows = env.observation_space.n
cols = env.action_space.n
Qtable = np.random.random((rows, cols))

NUM_EPISODES = 500
scores = []
for episode in range(NUM_EPISODES):  # Run for a certain number of games
    print(f'running episode {episode}')
    score = 0
    done = False
    state = env.reset()
    while not done:
        # Pick an action
        action = Qtable[state].argmax()
        # Perform the action
        next_state, reward, done, _ = env.step(action)
        # Update the Q table
        Qtable[state, action] = reward + gamma * Qtable[next_state].max()

        score += reward
        state = next_state

    scores.append(score)

plt.plot(scores, '.')
plt.show()