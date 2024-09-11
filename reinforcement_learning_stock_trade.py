import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

def get_data():
  df = pd.read_csv('/content/aapl_msi_sbux.csv')
  return df.values

class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size): # initializing our array buffers and pointers
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32) # storing the state
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32) # storing the next state
    self.acts_buf = np.zeros(size, dtype=np.unit8) # storing our actions
    self.rews_buf = np.zeros(size, dtype=np.float32) # stores our rewards
    self.done_buf = np.zeros(size, dtype=np.unit8) # stores the done flag
    self.ptr, self.size, self.max_size = 0, 0, size #  pointer starting at the value of 0, with current size of 0 and max size of "size argument"


  def store(self, obs, act, rew, next_obs, done): # stores our state, action rewards and flag
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size # incrementing the pointer, so that for the next time we run this func the values will be stored in the next position
    self.size = min(self.size+1, self.max_size) # when going to max size it returns to zeroo

  def smaple_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])

  def get_scalar(env):

    states = []
    for _ in range (env.n_step):
      action = np.random.choice(env.action_space)
      state, reward, done, info = env.step(action)
      states.append(state)
      if done:
        break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


  def maybe_make_dir(directory): # if a directory doesn't exist it creates it, cuz we're ging to store our trained model & rewards we encounter
    if not os.path.exists(directory):
      os.makedirs(directory)


  def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32): # creates a NN and returns it
    i = Input(shape=(input_dim,)) # input layer
    x = i

    for _ in range(n_hidden_layers):
      x= Dense(hidden_dim, activation='relu')(x) # hidden layers

    x = Dense(n_action)(x) # final layer

    model = Model(i, x) # model that will be trained by our agent

    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model

class MultiStockEnv: # environment class
# state -> vector f size 7: shares of 3 stocks + prices of 3 stocks + cash owned
# actionn -> categorical variable with 3^3 possibilities: 3 actions for each stock = sell, buy, hold


  def __init__(self, data, initial_investment=20000):
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    self.action_space = np.arange(3**self.n_stock)

    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock))) # a list of length 27 that gives examples of what we can have

    self.state_dim = self.n_stock * 2 + 1 # size of  state

    self.reset()


  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock) # setting to an array of zeros, that will tell us how many shares of each stock we own, which is 0 in the start
    self.stock_price = self.stock_price_history[self.cur_step] # price of stocks based on the day
    self.cash_in_hand = self.initial_investment
    return self._get_obs() # returning state


  def step(self, action): # preforms an action in the environment and returns the reward and next state
    assert action in self.action_space # check if the action actually exists in the action space

    prev_val = self._get_val() # getting the current value before action

    self.cur_step += 1 # updating the day
    self.stock_price = self.stock_price_history[self.cur_step] # updating the price (as the days go by)

    self._trade(action) # preforming the trade(a combination of buy, sell, hold on stocks)

    cur_val = self._get_val() # value after action

    reward = cur_val - prev_val # increase in portfolio value

    done = self.cur_step == self.n_step - 1 # checking if the data is over and setting the done flag

    info = {'cur_val': cur_val} # storing the current value of portfolio in the info dictionary

    return self._get_obs(), reward, done, info


  def _get_obs(self): # returns state
    obs = np.empty(self.state_dim)
    obs[:self.n_stock] = self.stock_owned # stocks we own, of size 3
    obs[self.n_stock:2*self.n_stock] = self.stock_price # value of stock, of size 3
    obs[-1] = self.cash_in_hand # cash in hand of our last observation
    return obs


  def _get_val(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand # number of shares * value of each stock + cash


  def _trade(self, action):
    action_vec = self.action_list[action] # 0=sell, 1=hold, 2=buy
    sell_index = [] # indices of stocks we wanna sell
    buy_index = [] # indices of stocks we wanna buy
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    if sell_index: # we sell all the shares of an stock, to simplify the problem

      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0

    if buy_index: # looping thru each stock we want to buy, buying 1 share at a time, until we can't buy anymore and we're out of cash

      can_buy = True
      while can_buy:
        for i in buy_index: # going thru all shares with different prices, cuz even thu if we can't buy some of them, but maybe we can buy the next ones
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1 # buying one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False

class DQNAgent(object): # describing the agent
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayBuffer(state_size, action_size, size=500) # memory size of 500
    self.gamma = 0.95 # discount rate
    self.epsilon = 1.0 # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = mlp(state_size, action_size) # getting an instance of our model by calling the mlp function


  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done) # storing the inputs of this function in replay buffer


  def act(self, state): # using epsilon greedy to choose an action
    if np.random.rand() <= self.epsilon: # generating a random number and comparing with epsilon
      return np.random.choice(self.action_size)
    act_values = self.model.predict(state) # getting all the Q values for the input of state
    return np.argmax(act_values[0])  # returns actions(which leads to the max Q value)


  def replay(self, batch_size=32): # function that does the learning
    if self.memory.size < batch_size: # if replay buffer contains enough data
      return

    minibatch = self.memory.sample_batch(batch_size) # sampling a batch of data from the replay memory, which returns a dictionary
    states = minibatch['s']
    actions = minibatch['a']
    rewards = minibatch['r']
    next_states = minibatch['s2']
    done = minibatch['d']

    target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1) # calculate the tentative target = Q(s', a)

    target[done] = rewards[done] # setting the target where the next state is terminal state to the reward, cuz the value of terminal state is zero

    target_full = self.model.predict(states) # settimng the target to prediction for all values so the target and predictions have the same value(as required in keras)
    # targets are 1D array here and predictions are a 2D array
    target_full[np.arange(batch_size), actions] = target # we only need the actions which were already taken

    self.model.train_on_batch(states, target_full) # run one training step

    if self.epsilon > self.epsilon_min: # updating epsilon to reduce amount of exploration
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)


  def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
      action = agent.act(state)
      next_state, reward, done, info = env.step(action)
      next_state = scaler.transform([next_state])
      if is_train == 'train':
        agent.update_replay_memory(state, action, reward, next_state, done)
        agent.replay(batch_size) # one step of gradient decsent
      state = next_states

    return info['cur_val']


  if __name__ == '__main__':
    models_folder = 'rl_tarder_models' # where to save ur models
    rewards_folder = 'rl_tarder_rewards' # where to save our rewards from both train & test phases
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000

    parser = argparse.ArgumentParser() # to run the script with command line raguments
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either"train" or "test"')
    args = parser.parse_args()

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data() # to get ur time series
    n_timesteps, n_stocks = data.shape

    # splitting the data into train & test
    n_train = n_timesteps // 2
    # first half is train, second half is test
    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment) # creating instance of environment object
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # store the final value of the portfolio (for each episode)
    portfolio_value = []

    if args.mode == 'test': # over writing things we have created in case we are in test mode
      with open(f'{models_folder}/scaler.pkl', 'rb') as f: # using the same scaler we had in training phsase
        scaler = pickle.load(f)

      env = MultiStockEnv(test_data, initial_investment)

      agent_epsilon = 0.01 # epsilon shouldn't be 1(which is pure exploration), also not 0(data and agent will be deterministic and we'll get the same results each time)

      agent.load(f'{models_folder}/dqn.h5') # loading the trained weights

    for e in range(num_episodes): # loop to play our episodes
      t0 = datetime.now()
      val = play_one_episode(agent, env, args.mode) # playinf an episode and getting the value of portfolio
      dt = datetime.now() - t0 # t know the duration of each iteration
      print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
      portfolio_value.append(val)

    if args.mode == 'train': # checking if the mode is train(we've finished playing all the episodes)
      agent.save(f"{models_folder}/dqn.h5") # save the weights when we're done

      with open(f'{models_folder}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f) # save the scaler

    # save the rewards
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

# to plot the rewards, so we know distributin of each reward during each episode
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'linear_rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

plt.hist(a, bins=20)
plt.title(args.mode)
plt.show()







