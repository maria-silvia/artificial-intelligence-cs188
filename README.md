# artificial-intelligence-cs188
Artificial Intelligence class assignments

From https://inst.eecs.berkeley.edu/~cs188/sp23/projects/ 

## Search
search/search.py

- DFS
- BFS
- UCS (Dijkstra)
- A*

##  Multiagent Search (games)
multiagent/multiAgents.py

- ReflexAgent
- MiniMax
- Alpha-Beta Pruning
- Expectimax
- Evaluation Function

## Reinforcement Learning

### MDP (Markov Decision Process)
reinforcement/valueIterationAgents.py

- Value Iteration: runValueIteration() 
- Policy Extraction: computeActionFromValues()
- computeQValueFromValues()

### Reinforcement Learning
reinforcement/qlearningAgents.py

- Q-Learning and Temporal difference learning: update()
    keeps updating the estimates of Q* values with using a learning rate (alpha) to weight

- Epsilon Greedy: getAction()
    at epsilon rate alternates between exploring something new (random action) or exploiting policy built so far

- Approximate Q-Learning and Feature-Based Representations: ApproximateQAgent
    Pacman is able to generalize, faster learning