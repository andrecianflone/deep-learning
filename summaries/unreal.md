Reinforcement Learning with Unsupervised Auxiliary Tasks, Jaderberg et al, ICLR 2017. [openreview](https://openreview.net/forum?id=SJ6yPD5xg). See Caruana PhD thesis from 1997, discusses auxiliary tasks for better representations!

The paper is about learning additional tasks, which don't require additional data, with the same parameters from the model which learns to maximize rewards with RL. This leads to better representations and performance, in score and speed!

A base agent learns to maximize rewards with A3C. Experiences are pushed to a replay buffer. Auxiliary tasks sample from this buffer for off-policy training. Interestingly, buffer sampling is actually intentionally skewed so samples are evenly split between rewarding and negative evnts.

The auxiliary tasks include:
- Pixel control: train agents that learn a separate policy for maximally changing the pixels in an input image. 
- Reward prediction: process a sequence of consecutive observations and require agent to predict the reward in following unseen frame
- Value function replay: resample recent historical sequences from behaviour policy distribution (via replay buffer) and perform extra value function regression in addition to the on-policy A3C. By resampling previous experience, and randomly varying the temporal position of the truncating window over which the n-step return is computed, value function replay performs value iteration and exploits newly discovered features shaped by reward prediction (sampling distribution not skewed in this case)
