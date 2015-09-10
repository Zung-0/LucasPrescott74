"""
filename: lucas_search_general.py
Author: Diego Zuniga and Alex Carrasco

Solves and simulates the general equilibirium of the 
Lucas and Prescott Search model for any economy by creating 
a list of islands as instances of the LucasSearch class
defined in lucas_search.

Following (1974) Equilibrium Search and Unemployment - Journal oF Economic Theory
by Robert E. Lucas, Jr. and Edward C. Prescott

"""

#Notes:
#Z: Check the note on get_lambda
#Z: I need to include the update method in the simulate method.
#	This will simplify the code.


from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from itertools import izip
from lucas_search import LucasSearch
from scipy.optimize import fminbound, brenth


class LucasSearchGeneral(object):
	'''
	Holds the specifications for an instance of Lucas and Prescott job search model
	(1974) Equilibrium Search and Unemployment - Journal oF Economic Theory

	Parameters
	----------
	beta : scalar(int), optional(default=.9)
		The utility discounting parameter
	states : array_like(int), optional(default=[1, 2])
		Contains the n possible states of demand
	transition : array_like(int), optional(default = [[.9, .1], [.1, .9]])
		Transition matrix for the demand states, should be an nxn matrix
	islands_wf : array_like(int), optional(default = [50]*100)
		Contains the workforce size of each island
	islands_state : array_like(int), optional(default = [0]*50 + [1]*50)
		Contains the state of demand of each island
	growth : scalar(float), optional(default = 0)
		Rate of growth of the economy's workforce
	stochastic : Boolean, optional(default = False)
		Whether the arrival process is stochastic or not
	Attributes
	----------
	grid : array_like(float, ndim=1)
		The grid over possible workforces.

	'''

	def __init__(self, beta=.9, states=[1, 2], transition=[[.9, .1], [.1, .9]], 
					islands_wf = [50]*100, islands_state = [0]*50 + [1]*50, 
					stochastic=False): 
	
		self.beta = beta
		self.states = np.asarray(states)
		self.transition = np.asarray(transition)
		self.islands_wf = np.asarray(islands_wf)
		self.islands_state = np.asarray(islands_state)
		self.population = self.islands_wf.sum()
		self.stochastic = stochastic

	def change_array(self, l):
		"""
		Calculates the the change in workforce
		for all the islands for an exogenously given
		value of search.
		
		Parameters
		----------
		l : scalar(float)
			Value of search. (lambda)
		
		Returns
		-------
		An array holding the change in workforce size for the whole
		economy.
		In equilibrium, its sum should be zero

		"""  
		
		wf_change = LucasSearch(beta=self.beta, states=self.states, transition=self.transition, lamb=l).workforce_change(stochastic=self.stochastic)
		return np.asarray([wf_change[state][y] for state, y in izip(self.islands_state, self.islands_wf)])	
	
	def net_moves(self, l):
		"""
		Calculates the sum of the changes in workforce
		for all the islands for an exogenously given
		value of search.
		
		Parameters
		----------
		l : scalar(float)
			Value of search. (lambda)
		
		Returns
		-------
		The net change in workforce size for the whole
		economy.
		In equilibrium, it should be zero.

		"""  
		wf_change = LucasSearch(beta=self.beta, states=self.states, transition=self.transition, lamb=l).workforce_change(stochastic=self.stochastic)
		return np.asarray([wf_change[state][y] for state, y in izip(self.islands_state, self.islands_wf)]).sum()
	

	def get_lambda(self):
		"""
		Computes the equilibrium value of lambda for the
		economy formed by the specified islands.

		Returns 
		-------
		lambda_star: scalar(float)
			Value of lambda that makes the workforce change
			of the complete economy 0. (Fixed workforce)
		"""
		#Z: My problem here is that in some cases there is no value of 
		#	lambda that makes net change equal 0-
		#	Perhaps I need to let go of the 'discrete number of workers 
		#	only' condition and start using non-integer values.
		#	Nevertheless, I am not sure I should lose too much sleep
		#	on that.

		change = lambda l: self.net_moves(l) 
		lambda_star = brenth(change, 0, 10)
		return lambda_star
	

	def update(self):
		"""
		Updates the workforce and state of demand of each island
		for the next period.
		"""
		
		l = self.get_lambda()
		
		change_wf = self.change_array(l)
		if self.stochastic:
			change_wf = self.change_array(l)*np.random.uniform(0,2, len(self.islands_wf))
		self.islands_wf = self.islands_wf + change_wf 
		change = np.random.choice((0,1), p=(.9, .1), size=len(self.islands_wf)) # 1 if the state changes, 0 otherwise
		self.islands_state = (self.islands_state + change)%2 # Computes the new state
 

	def simulate(self, T=100, compute_unemployment=False):
		"""
		Simulates the whole economy for T periods.

		Parameters
		----------
		T : scalar(int), optional(default=100)
			Number of periods to simulate.
		compute_unemployment : Boolean, optional(default=False)
			Whether or not to calculate the economy unemployment rate.

		Returns
		-------
		The net change in workforce size for the whole
		economy.
		In equilibrium, it should be zero.

		""" 
		Economy = self
		lambdas = np.empty(T)
		un_rate = np.empty(T)
		average_state = np.empty(T)

		for j in xrange(T):
			l = Economy.get_lambda()
			states = Economy.islands_state
			wf = Economy.islands_wf
			LS = LucasSearch(beta=self.beta, states=self.states, transition=self.transition, lamb=l)
						
			if compute_unemployment:
				employment = LS.compute_employment()
				unemployment = np.asarray([y - employment[state][list(LS.grid).index(y)] for state, y in izip(states, wf)])
				unemployment_rate = unemployment.sum()/np.asarray(wf).sum()
			
			lambdas[j] = l
			average_state[j] = np.asarray(states).mean()
			if compute_unemployment:
				un_rate[j] = unemployment_rate

			new_wf = LS.next_workforce()
			wf = [new_wf[state][list(LS.grid).index(y)] if y!= 0 else 0 for state, y in izip(states, wf)] 
			change = np.random.choice((0,1), p=(.9, .1), size=len(wf)) # 1 if the state changes, 0 otherwise
			states = (states + change)%2 # Computes the new state
			
			Economy = LucasSearchGeneral(beta=self.beta, states=self.states, transition=self.transition,
										islands_wf=wf, islands_state=states)

		if compute_unemployment:
			return lambdas, average_state, un_rate
		else:
			return lambdas, average_state