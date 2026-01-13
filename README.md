# Continuous Optimization for Feature Selection with Permutation-Invariant Embedding and Policy-Guided Search

## Basic info:
This is the release code for :
[Continuous Optimization for Feature Selection with Permutation-Invariant Embedding and Policy-Guided Search
](https://arxiv.org/abs/2505.11601) \(CAPS\)
which is accepted by KDD 2025!

![image](https://github.com/user-attachments/files/24596286/framework_2_bot.pdf)














***
## Paper Abstract
Feature selection removes redundant features to enhance both performance and computational efficiency in downstream tasks.
Existing methods often struggle to capture complex feature interactions and adapt to diverse scenarios.
Recent advances in this domain have incorporated generative intelligence to address these drawbacks by uncovering intricate relationships between features.
However, two key limitations remain: 1) embedding feature subsets in a continuous space is challenging due to permutation sensitivity, as changes in feature order can introduce biases and weaken the embedding learning process;
2) gradient-based search in the embedding space assumes convexity, which is rarely guaranteed, leading to reduced search effectiveness and suboptimal subsets.
To address these limitations, we propose a new framework that can: 1)  preserve feature subset knowledge in a continuous embedding space while ensuring permutation invariance; 2) effectively explore the embedding space without relying on strong convex assumptions.
For the first objective,  we develop an encoder-decoder paradigm to preserve feature selection knowledge into a continuous embedding space.
This paradigm captures feature interactions through pairwise relationships within the subset, removing the influence of feature order on the embedding. 
Moreover, an inducing point mechanism is introduced to accelerate pairwise relationship computations.
For the second objective, we employ a policy-based reinforcement learning  (RL) approach to guide the exploration of the embedding space.
The RL agent effectively navigates the space by balancing multiple objectives.
By prioritizing high-potential regions adaptively and eliminating the reliance on convexity assumptions, this search strategy effectively reduces the risk of converging to local optima.
Finally, we conduct extensive experiments to demonstrate the effectiveness, efficiency, robustness and explicitness of our model.
***
