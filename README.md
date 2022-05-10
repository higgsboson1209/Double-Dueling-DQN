# Double-Dueling-DQN
A Pytorch based custom implementation of a Double Dueling Deep Q Network which uses OpenAI Gym to train agents to play clasic control models like CartPole, LunarLander and more. 

## Items Implemented 
- **Double Q Learning Architecture**[[1]](#1)
  - Used to remove bias and minimize the reward maximization and over estimation problem\
- **Dueling Q Learning Architecture**
  ![Dueling DQN](https://github.com/higgsboson1209/Double-Dueling-DQN/blob/main/images/Dueling%20DQN.jpg) 
  - Dueling DQN breaks Q to an A (advantage) stream and a V (value )stream. Adding this type of structure to the network head allows the network to better differentiate actions from one another, and significantly improves the learning\.[[2]](#2)
- **Prioritized Experience Replay**[[3]](#3)
  - We implement a PER to enable the model to learn from samples that have the most amount of error for effective training, and shows great improvements over start replay buffer 
  - This implementation is powered by a sum tree implementation to ensure sampling and insertion of data is as effecient as possible (O(log(N) where N is the number of samples)[[4]](#4)

## Results
The agent gives great results on the cartpole environment and achieves a maximum reward of 500/500 in multiple episodes. With a maximum average reward of over 470\
![Graph of result](https://github.com/higgsboson1209/Double-Dueling-DQN/blob/main/images/performanceoncartpole.png)

## References
<a id="1">[1]</a> 
[Double DQN original paper](https://arxiv.org/pdf/1509.06461.pdf)\
<a id="2">[2]</a> 
[RL Coach by IntelLabs](https://intellabs.github.io/coach/components/agents/value_optimization/dueling_dqn.html)\
<a id="3">[3]</a> 
[How Prioritized Experience Replay works](https://danieltakeshi.github.io/2019/07/14/per/)\
<a id="4">[3]</a> 
[Blog explaining Sum Tree and how to Implement it](http://www.sefidian.com/2021/09/09/sumtree-data-structure-for-prioritized-experience-replay-per-explained-with-python-code/)\
