import csv
import math
import numpy as np
import matplotlib.pyplot as plt


def read_file(file_name):
	"""
	Read and process data to be used for clustering.
	:param file_name: name of the file containing the data
	:return: dictionary with element names as keys and feature vectors as values
	"""

	f = open(file_name, "rt", encoding="latin1")
	header = f.readline().strip().split(",")
	header = header[len(header)-59 : len(header)-12]
	data = {}

	for line in f:
		#print(line)
		row = line.strip().split(",")
		#print(row[len(row) - 13])
		#######fix str to int
		data[row[1], row[0]] = [str(x) for x in row[len(row) - 59 :len(row) - 12]]


	data = averageYears(data)

	newData = {}
	for i, h in enumerate(header):
		newData[h] = [data[key][i] for key in data]


	return newData

# povpreci tocke za vsako drzavo za vsa leta
# ko je drzava sodelovala na evroviziji
# v povprecje ne vkljuci nan vrednosti
def averageYears(data):

	newData = {}
	sumData = {}
	
	for key in data:
		#ce je drzava ze v tabeli se pristeje sestevku +1 drugace se ustvari z vrednostimi 1
		if key[0] in sumData:
			sumData[key[0]] = [x+1 for x in sumData[key[0]]]
		else:
			sumData[key[0]] = [1 for x in range(len(data[key]))]

		#vse vrednosti nan sprememni v 0
		for i, value in enumerate(data[key]):
			if value == '':
				data[key][i] = 0
				sumData[key[0]][i] = sumData[key[0]][i] - 1
			data[key][i] = int(data[key][i])
		#ce je drzava ze v novi tabeli podatke sesteje
		if key[0] in newData:
			tmp1 = data[key]
			tmp2 = newData[key[0]]
			newData[key[0]] = [tmp1[x]+tmp2[x] for x in range(len(tmp1))]

		#ce drzave se ni v novi tabeli jo doda
		else:
			newData[key[0]] = data[key]

	# da ustvari povprecje deli vrednosti s stevilom ponovitev	
	for key in newData:
		for i in range(len(newData[key])):
			if sumData[key][i] != 0:
				newData[key][i] = newData[key][i]/sumData[key][i]

	return newData


# spremeni vec dimenzionalno tabelo, ki vsebuje kljuce (imena drzav)
# v eno dimenzionalno tabelo
def flattenList(l):

	newList = []
	for i in l:
		if len(i) == 1:
			newList = np.append(newList, i)
		else:
			newList = np.append(newList, flattenList(i))

	return newList


# vrne indeks clustra v eno dimenzionalnem seznamu clustrov
def getIndex(cluster, clList):

	index = 0
	for i in clList:
		if i == cluster:
			return index
		index = index + 1
	return -1


# recurzivno izrise ascii dendrogram koncnih clustrov
def plotRekurzija(cl1, cl2, i):

	i = i + 1
	if len(cl1) == 1:
		for x in range(i):
			print("     ", end = "")
		print(" ---" + str(cl1[0]))
	else:
		plotRekurzija(cl1[0], cl1[1], i)

	for x in range(i - 1):
		print("     ", end = "")
	print("---|")

	
	if len(cl2) == 1:
		for x in range(i):
			print("     ", end = "")
		print(" ---" + str(cl2[0]))
	else:
		plotRekurzija(cl2[0], cl2[1], i)


# vrne tocki A in D, ki sta potrebni za izris clustrov
# z matplotlibom (ne za ascii izris)
def getAD(cl, distances, allConutries):

	if len(cl) == 1:
		x = 0
		y = getIndex(cl[0], allConutries)
	else:
		for i in distances:
			if cl[0] == i[0] and cl[1] == i[1]:
				x = i[2]

		tmp, ay = getAD(cl[0], distances, allConutries)
		tmp, dy = getAD(cl[1], distances, allConutries)

		y = (ay + dy)/2

	return x, y



class HierarchicalClustering:
	def __init__(self, data):
		"""Initialize the clustering"""
		self.data = data
		# self.clusters stores current clustering. It starts as a list of lists
		# of single elements, but then evolves into clusterings of the type
		# [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
		self.clusters = [[name] for name in self.data.keys()]


	def row_distance(self, r1, r2):
		"""
		Distance between two rows.
		Implement either Euclidean or Manhattan distance.
		Example call: self.row_distance("Polona", "Rajko")
		"""

		return math.sqrt(sum((a - b) ** 2
							 for a, b in zip(self.data[r1], self.data[r2])))


	def cluster_distance(self, c1, c2):
		"""
		Compute distance between two clusters.
		Implement either single, complete, or average linkage.
		Example call: self.cluster_distance(
			[[["Albert"], ["Branka"]], ["Cene"]],
			[["Nika"], ["Polona"]])
		"""

		if len(c1) > 1:
			rows1 = flattenList(c1)
		else:
			rows1 = c1

		if len(c2) > 1:
			rows2 = flattenList(c2)
		else:
			rows2 = c2


		#calculate the average distance
		allDistances = 0
		for i in rows1:
			for j in rows2:
				allDistances = allDistances + self.row_distance(i, j)

		allDistances = allDistances / (len(rows1) * len(rows2))

		return allDistances

	def closest_clusters(self):
		"""
		Find a pair of closest clusters and returns the pair of clusters and
		their distance.

		Example call: self.closest_clusters(self.clusters)
		"""
		clusters = self.clusters
		#calculate the smallest distance
		smallestDistance = float('inf')
		firstCluster = clusters[0]
		secondCluster = clusters[0]
		for indexi, i in enumerate(clusters):
			for indexj, j in enumerate(clusters):
				if indexj > indexi:
					distance = self.cluster_distance(i, j)
					if distance < smallestDistance:
						smallestDistance = distance
						firstCluster = i
						secondCluster = j

		return(firstCluster, secondCluster, smallestDistance)

	def run(self):
		"""
		Given the data in self.data, performs hierarchical clustering.
		Can use a while loop, iteratively modify self.clusters and store
		information on which clusters were merged and what was the distance.
		Store this later information into a suitable structure to be used
		for plotting of the hierarchical clustering.
		"""

		clusters = self.clusters
		distances = []
		while len(self.clusters) > 1:
			cl1, cl2, distance = self.closest_clusters()

			ind1 = getIndex(cl1, clusters)
			tmp1 = clusters.pop(ind1)
			ind2 = getIndex(cl2, clusters)
			tmp2 = clusters.pop(ind2)

			tmp = [tmp1, tmp2]

			clusters.insert(0, tmp)
			distances.insert(0, [cl1, cl2, distance])

		# ker ima tabela eno dimenzijo prevec
		self.clusters = self.clusters[0]


		# izris dendrigrama z matplotlib knjiznico
		allConutries = flattenList(distances[0][0])
		allConutries = np.append(allConutries, distances[0][1])
		yindex = [y for y in range(len(allConutries))]

		# za vsak cluster izrise tri crte:
		# ena crta je od prvega clustra do distance na x osi med obema
		# druga je od drugega clustra do distance na x osi med obema
		# tretja navpicno povecuje prejsnji crto
		for i in range(len(distances)):
			cl1, cl2, distance = distances[len(distances) - 1 - i]

			ax, ay = getAD(cl1, distances, allConutries)
			bx, by = distance, ay
			dx, dy = getAD(cl2, distances, allConutries)
			cx, cy = distance, dy

			x1, y1 = [ax, bx], [ay, by]
			x2, y2 = [dx, cx], [dy, cy]
			x3, y3 = [bx, cx], [by, cy]

			plt.plot(x1, y1, x2, y2, x3, y3, marker = '.')


		plt.yticks(yindex, allConutries)
		plt.tick_params(labelsize=8)
		plt.title('Glasovanje za pesem Evrovizije')
		axis = plt.gca()
		axis.set_xlim(11, 27)
		plt.show()



	def plot_tree(self):
		"""
		Use cluster information to plot an ASCII representation of the cluster
		tree.
		"""
		plotRekurzija(self.clusters[0], self.clusters[1], 0)

		pass


if __name__ == "__main__":
	DATA_FILE = "eurovision-final.csv"
	hc = HierarchicalClustering(read_file(DATA_FILE))
	hc.run()
	hc.plot_tree()