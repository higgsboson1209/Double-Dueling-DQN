import gym
import numpy as np
import torch
import copy
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from typing import NamedTuple

from agents.double_dueling_dqn import dueling_DQN_Network
from prioritized_experience_replay.memory import Memory

class Transition(NamedTuple):
    s: list
    a: float
    r: float
    s_p: list
    d: int
class DDQN:
   def __init__(self,environment,network,gamma=0.99,train_after=50000,train_freq=4,target_update=1000,batch_size=64,verbose=500,learning_rate=1e-4,max_grad_norm=10,tau=5e-3):
      self.environment=environment
      self.main_function=network
      self.target_function=copy.deepcopy(network)
      self.gamma = gamma
      self.train_after = train_after
      self.train_freq = train_freq
      self.target_update = target_update
      self.batch_size = batch_size
      self.optimizer = torch.optim.Adam(self.main_function.parameters(), lr=learning_rate)
      self.max_grad_norm = 10
      self.tau = tau
      self.EPS_lo=0.01
      self.memory_size=20000
      self.memory=Memory(self.memory_size)
      self.EPS=0.95
      self.EPS_decay=0.999
   
   def add_sample_to_memory(self,state,action,reward,next_state,done):
     
     s = torch.from_numpy(np.array(state)).type(torch.float32)
     a = torch.from_numpy(np.array(action)).type(torch.float32)
     s_p = torch.from_numpy(np.array(next_state)).type(torch.float32)

     if done==True:
       temp_done=1 
     else:
       temp_done=0
     r=reward
     q = self.main_function(s)[a.long()]
     with torch.no_grad():
         a_p = torch.argmax(self.main_function(s_p))  #actions selected from local q
         q_p = self.target_function(s_p)[a_p]    #actions evaluated from target ]
         y = r + self.gamma * q_p *(1-temp_done)
     error = abs(y-q)

     self.memory.add(float(error.item()), (state, action, reward, next_state, done))

   def update(self,batch,idxs,is_weights):
     s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
     a = torch.from_numpy(np.array(batch.a)).unsqueeze(1).type(torch.float32)
     r = torch.FloatTensor(batch.r).unsqueeze(1)
     s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
     done= torch.IntTensor(batch.d).unsqueeze(1)
     q = self.main_function(s).gather(1, a.long())
     with torch.no_grad():
         a_p = torch.argmax(self.main_function(s_p), dim = 1).unsqueeze(1)  #actions selected from local q
         q_p = self.target_function(s_p).gather(1, a_p)    #actions evaluated from target q
         y = r + self.gamma * q_p *(1-done)
     errors = torch.abs(y - q).data.numpy()
     for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
     loss = (torch.FloatTensor(is_weights) *F.mse_loss(q, y)).mean()
     self.optimizer.zero_grad()
     loss.backward()
     clip_grad_norm_(self.main_function.parameters(), self.max_grad_norm)
     self.optimizer.step()
     return loss
   def update_target_network(self):
     self.target_function= copy.deepcopy(self.main_function)
   def select_action(self, s):
     self.EPS = max(self.EPS_lo, self.EPS * self.EPS_decay)
     if torch.rand(1) > self.EPS:
         a = torch.argmax(self.main_function(torch.from_numpy(s).type(torch.float32)).detach()).numpy()
     else:
         a = self.environment.action_space.sample()
     return a 

def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n    #shape[0]
    #net = Q_val(obs_dim, act_dim)
    net = dueling_DQN_Network(obs_dim, act_dim)
    dqn_agent = DDQN(env, net, train_after=5, target_update=100, batch_size=32, verbose=2000, learning_rate=0.001)    #hyperparameters for lunarlander
    episodes = 0
    s_t = env.reset()

    episodic_rewards = deque(maxlen=20)

    for i in range(100000):
        a_t = dqn_agent.select_action(s_t)
        s_tp1, r_t, done, info = env.step(a_t)
        dqn_agent.add_sample_to_memory(s_t, a_t, r_t, s_tp1, done)
        s_t = s_tp1
        if dqn_agent.memory.tree.number_of_enteries>= dqn_agent.batch_size and i >= dqn_agent.train_after:
            
            if i % dqn_agent.train_freq == 0:
                sample=dqn_agent.memory.sample(dqn_agent.batch_size)
                batch=Transition(*zip(*sample[0]))
                dqn_agent.update(batch,sample[1],sample[2])

            if i % dqn_agent.target_update == 0:
                dqn_agent.update_target_network()
            
            if i % 2000== 0:
                avg_r = sum(episodic_rewards) / len(episodic_rewards)
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r} | Max. Reward: {max(episodic_rewards)}")

        if done:
            episodes += 1
            episodic_rewards.append(int(info["episode"]["r"]))
            s_t = env.reset()
    

    s_t = env.reset()
    #RENDERING THE TRAINED AGENT
    while True:
        env.render()
        a_t = dqn_agent.select_action(s_t)
        s_tp1, r_t, done, info = env.step(a_t)
        s_t = s_tp1 
        if done:
            print(f'Episode Complete, reward = {info["episode"]["r"]}')
            s_t = env.reset()
            return
if __name__ == "__main__":
  main()