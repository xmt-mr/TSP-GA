#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance as dis
from random import Random
import random
import math
import time

"""
GAによる解法
染色体：path representation
truncation selection
"""

class TSP:
	def __init__(self,path=None):
		""" 初期化を行う関数 """
		self.pop_size = 500
		self.survival_size = 250
		self.crossover_size = 225
		self.mutation_size = 25
		self.mutation_rate = 0.05

		if path is not None:
			self.loc = np.empty((0,2))
			with open(path) as f:
				for line in f.readlines():
					index,x,y = line.split(' ')
					self.loc = np.append(self.loc, [[int(x), int(y)]], axis=0)
			self.n_data = len(self.loc)						# 都市数
			self.dist = dis.squareform(dis.pdist(self.loc))	# 距離の表を作成
			#self.result = []			# もっともよかった順序を保存する
			self.best_fitness = None
		
	def fitness(self,order):
		n_order = len(order)
		f = np.sum( [ self.dist[order[i],order[(i+1)%n_order]] for i in np.arange(n_order) ] )
		return f
	
	def plot(self,order=None):
		""" 指定された順序でプロットする関数 """
		plt.clf()
		if order is None:
			plt.plot(self.loc[:,0],self.loc[:,1], '-o')
		else:
			plt.plot(self.loc[np.hstack((order,order[0])),0],self.loc[np.hstack((order,order[0])),1], '-o')
		plt.show()

	def new_plot(self,path=None,file=None):
		plt.clf()
		plt.plot(self.loc[:,0], self.loc[:,1], 'co')
		for _ in range(1, len(path)):
			i = path[_ - 1]
			j = path[_]
			# noinspection PyUnresolvedReferences
			plt.arrow(self.loc[i,0], self.loc[i,1], self.loc[j,0] - self.loc[i,0], self.loc[j,1] - self.loc[i,1], color='r', length_includes_head=True)
		i = path[0]
		j = path[-1]
		plt.arrow(self.loc[i,0], self.loc[i,1], self.loc[j,0] - self.loc[i,0], self.loc[j,1] - self.loc[i,1], color='r', length_includes_head=True)
		plt.xlim(-10, 110)
		plt.ylim(-10, 110)
		plt.axes().set_aspect('equal')
		plt.show()
		#plt.savefig(file)

	def pause(self,order=None):
		""" 指定された順序でリアルタイムプロットする関数 """
		plt.clf()
		if order is None:
			plt.plot(self.loc[:,0],self.loc[:,1], '-o')
		else:
			plt.plot(self.loc[np.hstack((order,order[0])),0],self.loc[np.hstack((order,order[0])),1], '-o')
		plt.xlim(-10, 110)
		plt.ylim(-10, 110)
		plt.axes().set_aspect('equal')
		plt.pause(0.0001)

	def solve(self):

		costs = np.empty((0,2))

		#初期解
		generation=0
		r = Random()
		population=[]
		for x in range(self.pop_size):
		    c=list(range(self.n_data))
		    r.shuffle(c)
		    population.append(c)

		while True:

			# sort
			population.sort(key=lambda x:self.fitness(x))

			# display best one
			#current_best_fitness = self.fitness(population[0])
			#if self.best_fitness != current_best_fitness:
			#	self.best_fitness = current_best_fitness
			self.pause(population[0])
			print("Generation ... %d , Cost ... %lf" % (generation, self.fitness(population[0])))
			costs = np.append(costs, [[generation, self.fitness(population[0])]], axis=0)

			# terminating condition
			if generation == 1000:
				break

			# remove m worst individuals
			population = population[:self.survival_size]

			# generate
			generation = generation+1
			new_population = []

			# crossover
			for k in range(self.crossover_size):
				parent1, parent2 = random.sample(population, k=2)

				p1 = random.randrange(self.n_data)
				p2 = random.randrange(self.n_data)
				while p1 == p2:
					p2 = random.randrange(self.n_data) # 同じ解にならないように

				if p1 > p2:
					p1, p2 = p2, p1

				child1 = parent1[p1:p2]
				child2 = parent2[p1:p2]
				length = p2 - p1

				p3 = p2
				while True:
					p3 = (p3+1) % self.n_data

					if parent1[p3] not in child2[:length+1]:
						child2.append(parent1[p3])
					if parent2[p3] not in child1[:length+1]:
						child1.append(parent2[p3])

					if p3 == p2:
						break

				new_population.append(child1)
				new_population.append(child2)


			# mutation
			for k in range(self.mutation_size):
				parent, = random.sample(population, k=1)
				child = parent[:]
				for i in range(self.n_data):
					if random.random() < self.mutation_rate:
						j = random.randrange(self.n_data)
						while i == j:
							j = random.randrange(self.n_data)
						child[i], child[j] = child[j], child[i]
				#if child not in population and child not in new_population:
				new_population.append(child)

				'''print(self.fitness(child))
					print(parent)
					print(child)'''


			population = population + new_population

		#print("Generation ... %d , Cost ... %lf" % (generation, self.fitness(population[0])))

		return population[0], costs


#@profile
def main():

	tsp = TSP(path='chn100.txt')
	result, costs = tsp.solve()
	tsp.new_plot(result)

if __name__=="__main__":
	main()
	
