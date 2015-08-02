"""
filename: lucas_search.py
Author: Diego Zuniga

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

		Av = [InterpolatedUnivariateSpline(self.grid, v[i], k=3) for i in xrange(len(self.states))]
		Avx = lambda y: np.asarray([function(y) for function in Av]) 

		case_a = self.lamb
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

	def compute_employment(self, v):
		Av = [InterpolatedUnivariateSpline(self.grid, v[i], k=3) for i in xrange(len(self.states))]
		Avx = lambda y: np.asarray([function(y) for function in Av])
		
		n_eq = np.asarray([np.empty(len(self.grid))]*len(self.states))
		
		function = [lambda n, s=state: self.labour_demand(n, s) + self.beta*np.dot(self.transition[j], Avx(n)) - self.lamb for j, state in enumerate(self.states)] 
		#function = [lambda n, s=state: self.labour_demand[j](n) + self.beta*np.dot(self.transition[j], Avx(n)) for j, state in enumerate(self.states)] 
			
		for j in xrange(len(self.states)):
			n_solve = fsolve(function[j], self.grid.max()/2) #unconstrained equilibrium labour
			n_eq[j] = np.ones(len(self.grid))*n_solve
			
		#Create the vector and check if the workforce constraint binds
		for i, y in enumerate(self.grid):
			for j in xrange(len(self.states)):
				n_eq[j][i] = min(n_eq[j][i], y)
		
		return n_eq



