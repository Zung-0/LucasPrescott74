"""
filename: lucas_search.py
Author: Diego Zuniga and Alex Carrasco

Solves (partially, for now) the Lucas and Prescott Search model

Following (1974) Equilibrium Search and Unemployment - Journal oF Economic Theory
by Robert E. Lucas, Jr. and Edward C. Prescott

"""

#Notes:

#Z: I think we don't need to do interpolation in the bellman operator. I'll simplify
#	the code later. 
#Z:	Also, there is probably a more Pythonic way to compute employment
#	like using n_eq[j, :]. Will check later.
#Z:	I still need to add a description for the methods of the class.

from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
#from compute_fp import compute_fixed_point


class LucasSearch(object):
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
	lambda : scalar(float), optional(default=3)
		The expected present value of searching for a job.
	
	Attributes
	----------
	grid : array_like(float, ndim=1)
		The grid over possible workforces.

	'''

	def __init__(self, beta=.9, states=[1, 2], transition=[[.9, .1], [.1, .9]], lamb=3):
		self.beta = beta
		self.states = states
		self.transition = transition
		#self.labour_demand = [lambda n, s=state: (s/2)*(1-.01*n) for state in states]
		self.labour_demand = lambda n, s: (s/2)*(1-.01*n)
		self.grid = np.linspace(1, 100, 100)
		self.lamb = lamb

	def bellman_operator(self, v):
		"""
		Bellman operator function updates a given
		value function defined on all the grid points.

		Parameters
		----------
		v : array_like(float, ndim=len(states))
			The value of the input function on different grid points
			for each of the possible states
		
		Returns
	    -------
    	Tv: array_like(float, ndim=len(states))
        	The updated value function

		"""
		#Interpolation of the value function
		##Z: I think interpolation is not really necessary for this. We could just iterate over 
		##	 the value function for each state. Like: for j, state in enumerate(states): for i, value in enumerate(v[j])
		Av = [InterpolatedUnivariateSpline(self.grid, v[i], k=3) for i in xrange(len(self.states))]
		Avx = lambda y: np.asarray([function(y) for function in Av]) 

		case_a = self.lamb

		#Array for holding the updated values
		Tv = np.asarray([np.empty(len(self.grid))]*len(self.states))
		for i, y in enumerate(self.grid):
			for j, state in enumerate(self.states):
				#case_b1 = self.labour_demand[j](y) + self.beta*self.transition[j][0]*Av[0](i) + self.beta*self.transition[j][1]*Av[1](i)
				#case_b1 = self.labour_demand(y, state) + self.beta*self.transition[j][0]*Av[0](i) + self.beta*self.transition[j][1]*Av[1](i)
				case_b1 = self.labour_demand(y, state) + self.beta*np.dot(self.transition[j], Avx(i))
				#case_b1 = self.labour_demand[j](y) + self.beta*np.dot(self.transition[j], Avx(i))
				#case_b2 = self.labour_demand[j](y) + self.lamb
				case_b2 = self.labour_demand(y, state) + self.lamb
				
				Tv[j][i] = max(case_a, min(case_b1, case_b2))

		return Tv

	def unconstrained_employment(self, v):
		"""
		Computes the (unconstrained) optimal level of employment
		in a market for every possible state of demand.  

		Parameters
		----------
		v : array_like(float, ndim=len(states))
			The value of the input function on different grid points
			for each of the possible states
		
		Returns
	    -------
    	n_hat: array_like(float, ndim=len(states))
        	The number of employed workers for each state of demand

		"""
		Av = [InterpolatedUnivariateSpline(self.grid, v[i], k=3) for i in xrange(len(self.states))]
		Avx = lambda y: np.asarray([function(y) for function in Av])
		
		n_hat = np.empty(len(self.states))
		
		#The zero in this function is the level of employment at which
		#workers are indifferent between leaving and staying.
		function = [lambda n, s=state: self.labour_demand(n, s) + 
					self.beta*np.dot(self.transition[j], Avx(n)) - 
					self.lamb for j, state in enumerate(self.states)] 
			
		for j in xrange(len(self.states)):
			n_solve = fsolve(function[j], self.grid.max()/2) #unconstrained equilibrium labour
			n_hat[j] = n_solve
			
		return n_hat

	def compute_employment(self, v):
		"""
		Computes the level of employment for every possible
		workforce and state of demand  

		Parameters
		----------
		v : array_like(float, ndim=len(states))
			The value of the input function on different grid points
			for each of the possible states
		
		Returns
	    -------
    	n_eq: array_like(float, ndim=len(states))
        	The number of employed workers for each workforce and
        	state of demand

		"""
		n_hat = self.unconstrained_employment(v)
		n_eq = np.asarray([np.empty(len(self.grid))]*len(self.states))
				
		#Imposes the workforce constraint if it is binding
		for j in xrange(len(self.states)):
			n_eq[j] = np.minimum(n_hat[j], self.grid)
					
		return n_eq

	def next_workforce(self, v, stochastic = False):
		"""
		Computes the next period workforce given current workforce
		and state of demand
		
		Parameters
		----------
		v : array_like(float, ndim=len(states))
			The approximate value function. After applying the
			bellman operator until convergence
		
		Returns
	    -------
    	next_wf: array_like(float, ndim=len(states))
        	The next period workforce for each current workforce 
        	and state of demand

		"""
		Av = [InterpolatedUnivariateSpline(self.grid, v[i], k=3) for i in xrange(len(self.states))]
		Avx = lambda y: np.asarray([function(y) for function in Av])
		
		n_eq = self.compute_employment(v)
		next_wf = np.asarray([np.empty(len(self.grid))]*len(self.states)) 

		if stochastic:
			x = np.linspace(0, 2, 100)
			fx = (1/len(x))*np.ones(len(x))
			def Avxx(y, a):
				""" 
				Computes the expected value of the value function using
				the predefined distribution.
				"""
				#Z: I should make this faster. No loops. Use arrays themselves. DONE
				values = np.asarray([np.empty(len(self.states))]*len(x)) 
				for i, x_val in enumerate(x):
					values[i] = Avx(y + a*x_val)*fx[i]
				return values.sum(axis = 0)
				
		for j, state in enumerate(self.states):
			for i, y in enumerate(self.grid):
				if n_eq[j][i] == y: # This means we are in case B1 or B2
					if stochastic:
						e_value =[self.beta*np.dot(self.transition[j], Avxx(y, a)) < self.lamb for a in xrange(100)] 
					else:
						e_value =[self.beta*np.dot(self.transition[j], Avx(y + a)) < self.lamb for a in xrange(100)]
					
					try:
						a_star = e_value.index(True) # This function gives the first a for which the function
					except ValueError:
						a_star = 0				 	 # self.beta*np.dot(self.transition[j], Avx(y + a)) - self.lamb
												 	 # becomes negative. Thus, it is the (expected if stochastic)
												 	 # number of workers arriving in the next period. 
					next_wf[j][i] = y + a_star
				
				else: # This is case A 
					next_wf[j][i] = next_wf[j][i-1] #As no more workers are being hired, 
													#next period workforce is the same as 
													#for a market with one less worker 	
									
		return next_wf