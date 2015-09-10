# LucasPrescott74
Lucas &amp; Prescott - Search Model

I wrote the Python code for solving this model. 

In a nutshell, this model is about workers that move through separate markets (it is easier to think of this markets as islands) searching for jobs. At any time, a worker is in an island and decides whether to stay and take a job in the island or go to another island searching for a job.

For a more detailed explanation, check either one of the notebooks.

The process of arriving to an island could be deterministic or stochastic. The notebook LucasPrescott74-NS (and the associated lucas_search_ns.py file) works only with a deterministic arrival process. Since the stochastic component can make things more messy, this notebook (and the code) is easier to read. 

The notebook LucasPrescott74 works with stochastic arrival. The associated Python file is lucas_search.py. This file contains the lucas_search_ns.py file but adds the stochastic arrival part.   

The code uses the bellman operator to compute a value function. This code also computes a Markov transition matrix to model the dynamics of the workforce.

The Python file lucas_search_general.py is provides a way of computing a general equilibrium when one of the assumptions is taken away (low number of islands). However, this code is not efficient and its usefulness is questionable.
