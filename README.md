# CasualNetworkProject
One of the main important task in the causal framework is the causal discovery task. The aim of which is to find the causal model underline some given observational data.

In particular, using the tool of causal network, the discovery task became the task of identifying:

a Directed Acyclic Graph (DAG) and
a set of conditional probability distribution.
These two subtasks are referred as:

structure learning
parameter learning
In order to justify the use of causal network we have to assume that the underlying causal process follows a probability distribution 
. So that, the underlying process can be represented by means of observational data sampled from the distribution 
.

In this notebook the focus will be on the activity of structure learning. Essentially, a partial implementation of the "Algorithm 1 Greedy Search" described in the paper [1] will be provided.
