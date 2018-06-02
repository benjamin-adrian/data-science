# -*- coding: utf-8 -*-

import World
import threading
import time
import random
import csv
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95	# discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.0001
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()
		plot_model(self.model, show_shapes=True, to_file='model.png')

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self):
		for state, action, reward, next_state, done in self.memory:
			#print(state, action, reward, next_state, done)
			target = reward
			if not done:
				target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] =  np.clip(target, -1.0, 1.0) # Perform clipping, as proposed by DQN
			World.set_cell_score(self.deflatten(state), World.actions[action], target)
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		self.memory.clear()

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

	def deflatten(self, s):
		t = s.argmax()
		return (t%World.x, int(t/World.x))
		
	def flatten(self, x, y):
		flat = y*World.x + x
		return flat

	def do(self, action):
		s = World.player
		r = -World.score
		if action == 0:
			World.try_move(0, -1)
		elif action == 1:
			World.try_move(0, 1)
		elif action == 2:
			World.try_move(-1, 0)
		elif action == 3:
			World.try_move(1, 0)
		else:
			return
		s2 = World.player
		r += World.score
		return s2, r, World.restart

		
	def run():
		state_size = World.x * World.y
		action_size = len(World.actions)
		agent = DQNAgent(state_size, action_size)
		# agent.load("./save/cartpole-dqn.h5")
		done = False
		EPISODES = 1000
		log = []
		score = 0		

		for e in range(EPISODES):
			done = False
			state = agent.flatten(World.player[0], World.player[1])
			state = np.reshape(np.identity(state_size)[state:state+1], [1, state_size])
			#print ("State: ", state)

			for time in range(20):
				c_reward = 0
				action = agent.act(state)
				next_state, reward, done = agent.do(action)
				c_reward += reward
				score += reward
				next_state = agent.flatten(World.player[0], World.player[1])
				next_state =  np.reshape(np.identity(state_size)[next_state:next_state+1], [1, state_size])
				agent.remember(state, action, c_reward, next_state, done)
				state = next_state
				if done:
					episode = {'episode': e, 'score': score, 'steps': time+1, 'alpha': agent.gamma, 'epsilon': agent.epsilon}
					log.append(episode)
					print(episode)
					break
			if not done:
				reward = -20
			agent.replay()		
			if e % 10 == 0:
				agent.save("./cartpole-dqn.h5")
			World.restart_game()	
		with open('data/log_eps-NN.csv', 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=['episode', 'score', 'steps', 'alpha', 'epsilon'])
			writer.writeheader()
			for episode in log:
				writer.writerow(episode)
		World.restart_game()
		


		
if __name__ == "__main__":
	actions = World.actions
	states = []
	Q = {}

	for i in range(World.x):
		for j in range(World.y):
			states.append((i, j))

	for state in states:
		temp = {}
		temp_e = {}
		for action in actions:
			temp[action] = 0.0 # Set to 0.1 if following greedy policy
			World.set_cell_score(state, action, temp[action])
		Q[state] = temp

	for (i, j, c, w) in World.specials:
		for action in actions:
			Q[(i, j)][action] = w
			World.set_cell_score((i, j), action, w)

	t = threading.Thread(target=DQNAgent.run)
	t.daemon = True
	t.start()
	World.start_game()
