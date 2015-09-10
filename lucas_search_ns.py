"""
filename: lucas_search.py
Author: Diego Zuniga and Alex Carrasco

Solves the Lucas and Prescott Search model. 
*Deterministic arrival process

Following (1974) Equilibrium Search and Unemployment - Journal oF Economic Theory
by Robert E. Lucas, Jr. and Edward C. Prescott

"""

#Z: I am making this version because the stochastic arrival process is somewhat messy.

from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve, brenth
from compute_fp import compute_fixed_point
from itertools import izip
from flatten import flatten
from scipy.sparse import lil_matrix, coo_matrix

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
		self.states = np.asarray(states)
		self.transition = np.asarray(transition)
		self.labour_demand = lambda n, s: (s/2)*(1-.01*n)
		self.grid = np.linspace(0, 200, 201)
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
		Av = [InterpolatedUnivariateSpline(self.grid, v[i], k=3) for i in xrange(self.states.size)]
		Avx = lambda y: np.asarray([function(y) for function in Av]) 

		case_a = self.lamb

		#Array for holding the updated values
		Tv = np.asarray([np.empty(self.grid.size)]*self.states.size)
		for i, y in enumerate(self.grid):
			for j, state in enumerate(self.states):
				case_b1 = self.labour_demand(y, state) + self.beta*np.dot(self.transition[j], Avx(i))
				case_b2 = self.labour_demand(y, state) + self.lamb
				
				Tv[j][i] = max(case_a, min(case_b1, case_b2))

		return Tv

	def unconstrained_employment(self, v=None):
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
		if v == None:
			v = np.asarray([np.zeros(self.grid.size)]*self.states.size)
			v = compute_fixed_point(self.bellman_operator, v, verbose=0, max_iter=30)

		Av = [InterpolatedUnivariateSpline(self.grid, v[i], k=3) for i in xrange(self.states.size)]
		Avx = lambda y: np.asarray([function(y) for function in Av])
		
		#The zero in this function is the level of employment at which
		#workers are indifferent between leaving and staying.
		function = [lambda n, s=state: self.labour_demand(n, s) + 
					self.beta*np.dot(self.transition[j], Avx(n)) - 
					self.lamb for j, state in enumerate(self.states)] 
		
		n_hat = [int(fsolve(function[j], self.grid.max()/2)) for j in xrange(self.states.size)]
		return n_hat

	def compute_employment(self, v=None):
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
		if v == None:
			v = np.asarray([np.zeros(self.grid.size)]*self.states.size)
			v = compute_fixed_point(self.bellman_operator, v, verbose=0, max_iter=30)

		n_hat = self.unconstrained_employment(v)
		n_eq = np.asarray([np.empty(self.grid.size)]*self.states.size)
				
		#Imposes the workforce constraint if it is binding
		for j in xrange(self.states.size):
			n_eq[j] = np.maximum(0,np.minimum(n_hat[j], self.grid))
					
		return n_eq


	def next_workforce(self, v=None):
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
		if v == None:
			v = np.asarray([np.zeros(self.grid.size)]*self.states.size)
			v = compute_fixed_point(self.bellman_operator, v, verbose=0, max_iter=30)
		
		Av = [InterpolatedUnivariateSpline(self.grid, v[i], k=3) for i in xrange(self.states.size)]
		Avx = lambda y: np.asarray([function(y) for function in Av])
		
		n_eq = self.compute_employment(v)
		next_wf = np.asarray([np.empty(self.grid.size)]*self.states.size) 

		for j, state in enumerate(self.states):
			for i, y in enumerate(self.grid):
				if n_eq[j][i] == y: # This means we are in case B1 or B2
					e_value =[self.beta*np.dot(self.transition[j], Avx(y + a)) < self.lamb for a in xrange(100)]
					try:
						a_star = e_value.index(True) # This function gives the first a for which the function
					except ValueError:
						a_star = 0				 	 # self.beta*np.dot(self.transition[j], Avx(y + a)) - self.lamb
													 # becomes negative. Thus, it is the  number of workers 
													 # arriving in the next period. 
					next_wf[j][i] = y + a_star
				
				else: # This is case A 
					if i == 0:
						next_wf[j][i] = n_eq[j][i]
					else:
						next_wf[j][i] = next_wf[j][i-1] #As no more workers are being hired, 
													#next period workforce is the same as 
													#for a market with one less worker 	
									
		return next_wf

	def workforce_change(self, v=None):
		"""
		Computes the change in workforce size given current
		workforce and state of demand
		
		Parameters
		----------
		v : array_like(float, ndim=len(states))
			The approximate value function. After applying the
			bellman operator until convergence
		
		Returns
		-------
		wf_change: array_like(float, ndim=len(states))
			The next period workforce for each current workforce 
			and state of demand

		"""
		if v == None:
			v = np.asarray([np.zeros(self.grid.size)]*self.states.size)
			v = compute_fixed_point(self.bellman_operator, v, verbose=0, max_iter=30)
		
		return self.next_workforce(v) - self.grid

	def markov_transition(self, v=None):
		"""
		Creates the transition matrix for workforce sizes 
		and states of demand.
		"""
		if v == None:
			v = np.asarray([np.zeros(self.grid.size)]*self.states.size)
			v = compute_fixed_point(self.bellman_operator, v, verbose=0, max_iter=30)
		
		new_wf = self.next_workforce(v)
		pll, plh, phl, phh = self.transition.flatten()
		
		I = np.hstack((self.grid, self.grid, self.grid.size + self.grid, self.grid.size + self.grid))
		J = np.hstack((new_wf[0], self.grid.size + new_wf[0], new_wf[1], self.grid.size + new_wf[1]))
		V = np.hstack((pll*np.ones(self.grid.size), plh*np.ones(self.grid.size), 
						phl*np.ones(self.grid.size), phh*np.ones(self.grid.size)))
	
		A = coo_matrix((V,(I,J)),shape=(self.states.size*self.grid.size, self.states.size*self.grid.size))

		M = A.tocsr().transpose()

		return M

	def compute_eq(self, M, vec=None, compute_unemployment=True, error_tol=1e-6, max_iter = 1000):
		"""
		Computes the equilibrium distribution of workforce sizes and
		states of demand as the stationary distribution of the 
		Markov transition matrix.
		
		Parameters
		----------
		M : sparse matrix (array_like?) (float, ndim=(states.size*grid.size)**2)
			The transition matrix for the parameter values
		vec : array_like (float, ndim= 1x(states.size*grid.size))
			Initial distribution on workforce size and state of demand
		error_tol : scalar (float)
			Error tolerance 
		max_iter : scalar (int)
			Maximum number of iterations

		Returns
		-------
		V : array_like (float, ndim= 1x(states.size*grid.size))
			Stationary distribution over states.
		"""

		if vec == None:
			vec = np.ones(self.states.size*self.grid.size)/(self.states.size*self.grid.size)

		error = 1
		iter = 1
		V = vec
		while error > error_tol and iter < max_iter:
			new_V = M.dot(V)
			error = np.max(np.abs(V-new_V))
			V = new_V
			iter += 1

		if compute_unemployment:
			employment = self.compute_employment()
			unemployment = np.hstack((self.grid, self.grid)) - employment.flatten() 
			un_rate = unemployment / np.hstack((self.grid, self.grid))
			unem = V.dot(unemployment)
			
			return V, unem	
		return V


	def get_lambda(self, workforce_size):
		"""
		Returns the value of lambda for a given average workforce size

		Parameters
		----------
		workforce_size : scalar (int)
			Average workforce size of the economy.

		Returns
		-------
		lambda_star : scalar(float)
			The value of search that would make the average workforce 
			per island be equal to 'workforce_size'
		"""
		
		function = lambda lamb: self.avg_workforce(lamb) - workforce_size

		lambda_star = fsolve(function, self.lamb)

		return lambda_star
