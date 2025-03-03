# 📖 Reinforcement Learning

### **Description**  
This section covers **Markov Decision Processes (MDP)**, **Q-learning for tabular reinforcement learning**, **Deep Q-Networks (DQN) for deep reinforcement learning**, **policy gradient methods (REINFORCE, PPO)**, and **actor-critic architectures (A2C)**.

---

## ✅ **Checklist & Key Considerations**  

- ✅ **Markov Decision Processes (MDP)**  
  - Define **states, actions, rewards, and transition probabilities**.  
  - Ensure the **Bellman equation holds** for optimal policy computation.  

- ✅ **Q-Learning Implementation**  
  - Train a **Q-table** (`numpy.zeros()`) to learn optimal policies.  
  - Implement **epsilon-greedy policy** for exploration-exploitation trade-off.  
  - Use **discount factor (γ) balancing future rewards** to prevent short-sighted behavior.  

- ✅ **Deep Q-Network (DQN) Implementation**  
  - Use **neural networks** to approximate Q-values for large state spaces.  
  - Optimize Q-learning updates with **experience replay** and **target networks**.  
  - Handle **unstable learning** by adjusting **learning rates and replay buffer sizes**.  

- ✅ **Policy Gradient Methods**  
  - Implement **Policy Networks** (`torch.nn.Linear`) for learning action distributions.  
  - Train models using **REINFORCE algorithm** with gradient ascent.  
  - Regularize policy gradients with **entropy loss** to encourage exploration.  

- ✅ **Proximal Policy Optimization (PPO)**  
  - Use **PPO with clipped objective functions** to prevent policy divergence.  
  - Balance **policy updates with KL divergence constraints**.  
  - Optimize training stability with **multiple epochs of minibatch training**.  

- ✅ **Actor-Critic (A2C) Implementation**  
  - Implement **two separate networks (actor & critic)** to stabilize learning.  
  - Train **advantage functions** to reduce variance in policy gradients.  
  - Use **Advantage Actor-Critic (A2C)** as a baseline for modern RL architectures.  
