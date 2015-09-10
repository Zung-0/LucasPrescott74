"""
filename: lucas_search.py
Author: Diego Zuniga and Alex Carrasco

Solves the Lucas and Prescott Search model.

Following (1974) Equilibrium Search and Unemployment - Journal oF Economic Theory
by Robert E. Lucas, Jr. and Edward C. Prescott

"""

#Notes:

#Z: I think interpolation is not really necessary for this. We could just iterate over 
#	the value function for each state. Like: for j, state in enumerate(states): for i, value in enumerate(v[j])
# 	I'll simplify the code later. 

#Z: I don't like the code for the transition matrix in the stochastic case. Nevertheless, it works.
#	I'll think of some easier way to do it.

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

	def next_workforce(self, v=None, stochastic = False):
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

		if stochastic:
			x = np.linspace(0, 2, 100)
			fx = (1/len(x))*np.ones(len(x))
			def Avxx(y, a):
				""" 
				Computes the expected value of the value function using
				the predefined distribution.
				"""
				values = [Avx(y + a*x_val)*fx_val for x_val, fx_val in izip(x, fx)] 
				return np.asarray(values).sum(axis=0)
				
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
					if i == 0:
						next_wf[j][i] = n_eq[j][i]
					else:
						next_wf[j][i] = next_wf[j][i-1] #As no more workers are being hired, 
													#next period workforce is the same as 
													#for a market with one less worker 	
									
		return next_wf

	def workforce_change(self, v=None, stochastic=False):
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
		
		return self.next_workforce(v=v, stochastic=stochastic) - self.grid

	def markov_transition(self, v=None, stochastic=False):
		"""
		Creates the transition matrix for workforce sizes 
		and states of demand.
		"""
		
		if v == None:
			v = np.asarray([np.zeros(self.grid.size)]*self.states.size)
			v = compute_fixed_point(self.bellman_operator, v, verbose=0, max_iter=30)
		

		new_wf = self.next_workforce(v, stochastic=stochastic)
		pll, plh, phl, phh = self.transition.flatten()
		
		if stochastic:
			bottom = 0
			top = 2

			incoming = np.asarray([[(new_wf[state-1][y] - y)*(new_wf[state-1][y] > y) 
									for y in self.grid] for state in self.states], dtype=int)
									
			inc_index_0 = [[i for y_1 in xrange(int(bottom*y), int(top*y + 1))] 
									for i, y in enumerate(incoming[0]) if abs(y) >.1]
			inc_col_0 = [[i + y_1 for y_1 in xrange(int(bottom*y), int(top*y + 1))] 
									for i, y in enumerate(incoming[0]) if abs(y) >.1]
			inc_val_0 = [[1./(int(top*y+1)-int(bottom*y)) if y_1!=y else 1./(int(top*y+1)-int(bottom*y)) - 1 
									for y_1 in xrange(int(bottom*y), int(top*y + 1))] 
									for i, y in enumerate(incoming[0]) if abs(y) >.1]
									
			inc_index_1 = [[i for y_1 in xrange(int(bottom*y), int(top*y + 1))] 
									for i, y in enumerate(incoming[1]) if abs(y) >.1]
			inc_col_1 = [[i + y_1 for y_1 in xrange(int(bottom*y), int(top*y + 1))] 
									for i, y in enumerate(incoming[1]) if abs(y) >.1]
			inc_val_1 = [[1./(int(top*y+1)-int(bottom*y)) if y_1!=y else 1./(int(top*y+1)-int(bottom*y)) - 1 
									for y_1 in xrange(int(bottom*y), int(top*y + 1))] 
									for i, y in enumerate(incoming[1]) if abs(y) >.1]
			
			inc_index_0 = np.asarray(flatten(inc_index_0))
			inc_index_1 = np.asarray(flatten(inc_index_1))
			inc_val_0 = np.asarray(flatten(inc_val_0))
			inc_val_1 = np.asarray(flatten(inc_val_1))
			inc_col_0 = np.asarray(flatten(inc_col_0))
			inc_col_1 = np.asarray(flatten(inc_col_1))

			incoming_ind = np.hstack((inc_index_0, inc_index_0, self.grid.size + inc_index_1, self.grid.size + inc_index_1))
			incoming_col = np.hstack((inc_col_0, self.grid.size + inc_col_0, inc_col_1, self.grid.size + inc_col_1))
			incoming_val = np.hstack((pll*inc_val_0, plh*inc_val_0, phl*inc_val_1, phh*inc_val_1))
		 
			I = np.hstack((self.grid, self.grid, self.grid.size + self.grid, self.grid.size + self.grid, incoming_ind))
			J = np.hstack((new_wf[0], self.grid.size + new_wf[0], new_wf[1], self.grid.size + new_wf[1], incoming_col))
			V = np.hstack((pll*np.ones(self.grid.size), plh*np.ones(self.grid.size), phl*np.ones(self.grid.size), phh*np.ones(self.grid.size), incoming_val))

		else:
			I = np.hstack((self.grid, self.grid, self.grid.size + self.grid, self.grid.size + self.grid))
			J = np.hstack((new_wf[0], self.grid.size + new_wf[0], new_wf[1], self.grid.size + new_wf[1]))
			V = np.hstack((pll*np.ones(self.grid.size), plh*np.ones(self.grid.size), phl*np.ones(self.grid.size), phh*np.ones(self.grid.size)))
		

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
			unemployment = np.tile(self.grid, self.states.size) - employment.flatten() 
			#un_rate = unemployment / np.tile(self.grid, self.states.size)
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
