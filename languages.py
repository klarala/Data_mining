from unidecode import unidecode
import glob
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# makes a dictionary of three letters and their relative frequency
def kmers(s, k=3):
    new = {}
    for i in range(len(s)-k+1):
        if s[i:i+k] not in new:
        	new[s[i:i+k]] = 1
        else:
        	new[s[i:i+k]] = new[s[i:i+k]] + 1

    # normalize the frequencies
    for i in new:
        new[i] = new[i]/len(new)

    return new

def readFile(path):
	corpus = {}
	for fileName in glob.glob(path):
	    name = os.path.splitext(os.path.basename(fileName))[0]
	    f = open(fileName, "rt", encoding="utf8").readlines()
	    text = " ".join([line.strip() for line in f])
	    text = text.lower()
	    corpus[name] = unidecode(text)


	# for each language, split it into triples and count
	# frequencies of each triplet
	for key in corpus:
		corpus[key] = kmers(corpus[key])

	return corpus

# returns dot distnace between vectors v1 and v2
def dotDistance(v1, v2):

	distance = 0

	for i in range(len(v1)):
		distance = distance + v1[i]*v2[i]

	return distance

# returns magnitude of vector v
def magnitude(v):

	magnitude = 0

	for i in range(len(v)):
		magnitude = magnitude + v[i]**2

	return math.sqrt(magnitude)

# returns cosine similarity between two vectors
def cosineSimilarity(v1, v2):

	cosDis = dotDistance(v1, v2) / (magnitude(v1)*magnitude(v2))

	return cosDis

# to measure cosine similarity between two languages, two vectors with all the possible
# combinations of three letters in the two languages are created
def distance (lan1, lan2):

	allTriples = list(set(lan1.keys() | lan2.keys()))
	
	newLan1 = {}
	newLan2 = {}

	for triplet in allTriples:
		newLan1[triplet] = lan1[triplet] if triplet in lan1 else 0
		newLan2[triplet] = lan2[triplet] if triplet in lan2 else 0


	newLan1 = list(newLan1.values())
	newLan2 = list(newLan2.values())

	distance = cosineSimilarity(newLan1, newLan2)

	return distance

# returns cost of the cluster
def calculateCost(clusters, distances):

	cost = 0

	for medoid in clusters:
		for language in clusters[medoid]:
			# needed so this function can be used for both the clusters to which
			# the element belongs and also a neighbour cluster
			if language != medoid:
				# the closer the languages are, closer to 1 distance is
				# that is why we use 1 - distance[lang][el]
				cost = cost + 1 - distances[language][medoid]

	return cost

# asociates each language to the closest of the 5 medoids
# passed into the function as clusters
def asociateLang(clusters, distances):

	for language in distances:
		closestDistance = 0.0
		closestMedoid = list(clusters.keys())[0]
		for medoid in clusters.keys():
			distance = distances[language][medoid]
			if distance > closestDistance:
				closestDistance = distance
				closestMedoid = medoid
		clusters[closestMedoid].append(language)

	return clusters

# returns average distance between language and cluster
def avgDistance(language, cluster, distances):

	distance = 0
	# counter is used because some clusters contain itself
	counter = 0
	for element in cluster:
		# needed so this function can be used for both the clusters to which
		# the element belongs and also a neighbour cluster
		if element != language:
			# the closer the languages are, closer to 1 distance is
			# that is why we use 1 - distance[lang][el]
			distance = distance + 1 - distances[language][element]
			counter = counter + 1


	return distance/counter if counter != 0 else 0

def calculateDistances(corpus):

	distances = {}
	for i, key1 in enumerate(corpus):
		for j, key2 in enumerate(corpus):
			if j <= i:
				dist = distance(corpus[key1], corpus[key2])
				if key1 in distances:
					distances[key1].update({key2: dist})
				else:
					distances[key1] = {key2: dist}
				if key2 in distances:
					distances[key2].update({key1: dist})
				else:
					distances[key2] = {key1: dist}

	return distances

def drawHistogram(silhouette, clusters, avgSil, epochs):
	# draw the histogram of silhouettes
	# x and y values definitons
	xData = list(silhouette.values())
	yData = list(silhouette.keys())
	# five colours for five clusters
	colours = ['gold', 'darkorange', 'firebrick', 'yellowgreen', 'darkgreen']
	fig = plt.figure()
	# for each language horizontal bar of histogram is drawn
	for i, medoid in enumerate(clusters):
		for j, language in enumerate(clusters[medoid]):
			plt.barh(list(silhouette.keys()).index(language), width = silhouette[language], color = colours[i])
	# assigment of y values
	plt.yticks(np.arange(22), yData);
	plt.tick_params(labelsize=7)
	plt.title('average silhouette = ' + str(avgSil))
	#plt.show()
	#fig.savefig('plot/plot' + str(epochs + 1) + '.png')
	plt.close()

def kClustering(corpus, distances):
	# the clustering is done 100 times, because we get different clusters
	# depending on which languages as medoids we start with
	averageSilhouettes = []
	for epochs in range(100):
		#select 5 random medoids
		allLanguages = list(corpus.keys())
		medoids = random.sample(allLanguages, 5)

		# dictionary with medoids as keys is created
		clusters = {}
		for i in range(5):
			clusters[medoids[i]] = []

		# languages are assigned to closest medoid
		clusters = asociateLang(clusters, distances)

		#calculate the cost
		cost = 100
		newCost = calculateCost(clusters, distances)


		#iterate while the cost decreases
		while newCost < cost:
			# In each cluster, make the point that minimizes the sum of distances within the cluster the medoid
			newClusters = {}
			for medoid in clusters:
				minimalMedoid = medoid
				minimalCost = calculateCost({minimalMedoid: clusters[medoid]}, distances)
				for language in clusters[medoid]:
					tmpMedoid = language
					tmpCost = calculateCost({tmpMedoid: clusters[medoid]}, distances)
					if tmpCost < minimalCost:
						minimalCost = tmpCost
						minimalMedoid = tmpMedoid
			
				# medoid that minimizes the sum of distances is key in new clusters
				newClusters[minimalMedoid] = []

			# Reassign each point to the cluster defined by the closest medoid determined in the previous step.
			newClusters = asociateLang(newClusters, distances)
			cost = newCost
			newCost = calculateCost(newClusters, distances)
			# new clusters are saved only if cost is smaller
			if newCost < cost:
				clusters = newClusters


		# CALCULATE THE SILHOUETTE
		silhouette = {}
		for medoid in clusters:
			for language in clusters[medoid]:
				# calculate the average distance to data points of neighbouring clouster
				neighbDist = 100
				for medoid2 in clusters:
					if medoid2 != medoid:
						tmpDistance = avgDistance(language, clusters[medoid2], distances)
						if tmpDistance < neighbDist:
							neighbDist = tmpDistance

				# calculate the average distance to data points of assigned cluster
				assDistance = avgDistance(language, clusters[medoid], distances)

				# calculate the silhouette and save it to dictionary silhouette
				if max(neighbDist, assDistance) != 0.0:
					silhouette[language] = (neighbDist - assDistance)/max(neighbDist, assDistance)
				else:
					silhouette[language] = 0.0
				if len(clusters[medoid]) == 1:
					silhouette[language] = 0

		#calculate the average silhouette and save it to list
		avgSil = 0
		for sil in silhouette:
			avgSil = avgSil + silhouette[sil]
		averageSilhouettes.append(avgSil/len(silhouette))

		# draws histogram of silhouettes and saves them to folder plot
		#drawHistogram(silhouette, clusters, avgSil/len(silhouette), epochs)
	return averageSilhouettes

def drawAvgSil(averageSilhouettes):
	# draw the histogram of silhouettes
	# x and y values definitons
	xData = np.arange(100)
	yData = averageSilhouettes
	fig = plt.figure()
	# for each silhouette bar of histogram is drawn
	plt.bar(xData, height = averageSilhouettes, color = 'darkcyan')
	plt.xlabel('index')
	plt.ylabel('average silhouette')
	plt.title('average silhouettes')
	plt.show()
	#fig.savefig('plot/avgSilhouettes.png')
	plt.close()

def langRecognition(corpus, path):
	results = {}
	# read text files and save data in dictionary
	lyrics = {}
	for fileName in glob.glob(path):
	    name = os.path.splitext(os.path.basename(fileName))[0]
	    f = open(fileName, "rt", encoding="utf8").readlines()
	    text = " ".join([line.strip() for line in f])
	    text = text.lower()
	    lyrics[name] = unidecode(text)

	# for each song, split it into triples and count
	# frequencies of each triplet
	for song in lyrics:
		lyrics[song] = kmers(lyrics[song])

	for song in lyrics:
		# find distance to which languages is the smallest
		songLang = {}
		for language in corpus:
			tmpDist = distance(corpus[language], lyrics[song])
			songLang[language] = tmpDist

		threeLang = []
		for i in range(3):
			#find maximum
			maximum = 0.0
			maxLang = ''
			for language in songLang:
				if songLang[language] > maximum:
					maximum = songLang[language]
					maxLang = language
			songLang.pop(maxLang)
			threeLang.append((maxLang, maximum))

		results[song] = threeLang

	# draw the histogram of silhouettes
	# y values definitons
	yData = list(results.keys())
	newYData = []
	for i in range(len(yData)):
		newYData.append('')
		newYData.append(yData[i])
		newYData.append('')

	# eleven colours for eleven songs
	colours = ['gold', 'orange', 'red', 'firebrick', 'yellowgreen', 'olive', 'black']
	fig = plt.figure()
	# for each language horizontal bar of histogram is drawn
	for i, song in enumerate(results):
		for j, language in enumerate(results[song]):
			bar, = plt.barh((i*3 + j), width = language[1], color = colours[i%len(colours)])
			# write language next to bar
			height = bar.get_height()
			plt.text(language[1], bar.get_y(), '%s' % language[0], size = 6)

	# assigment of y values
	plt.yticks(np.arange(len(newYData)), newYData);
	plt.tick_params(labelsize=7)
	plt.title('language recognition')
	plt.show()
	#fig.savefig('plot/plot' + str(epochs + 1) + '.png')
	plt.close()

	return(results)

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
		self.clusters = [[name] for name in self.data.keys()]


	def cluster_distance(self, c1, c2, distances):
		"""
		Compute distance between two clusters.
		Implement either single, complete, or average linkage.
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
				allDistances = allDistances + 1 - distances[i][j]

		allDistances = allDistances / (len(rows1) * len(rows2))

		return allDistances

	def closest_clusters(self, distances):
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
					distance = self.cluster_distance(i, j, distances)
					if distance < smallestDistance:
						smallestDistance = distance
						firstCluster = i
						secondCluster = j

		return(firstCluster, secondCluster, smallestDistance)

	def run(self, distancesDict):
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
			cl1, cl2, distance = self.closest_clusters(distancesDict)

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
		tmp = flattenList(distances[0][1])
		allConutries = np.append(allConutries, tmp)
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
		plt.tick_params(labelsize=7)
		plt.title('Podobnost jezikov')
		plt.show()




if __name__ == '__main__':

	# read text files and save data in corpus
	corpus = readFile("data/*.txt")
	
	# save all the distnaces into dictionary called distances
	# so that we don't need to recalculate them each time
	distances = calculateDistances(corpus)

	# CLUSETRING
	averageSilhouettes = kClustering(corpus, distances)
	#draw average silhouettes
	drawAvgSil(averageSilhouettes)

	# LANGUAGE RECOGNITION
	langRecognition(corpus, "lyrics/*.txt")

	#HIERARHICAL CLUSTERING
	hc = HierarchicalClustering(corpus)
	hc.run(distances)

	# NEWS RECOGNITION
	langRecognition(corpus, "news/*.txt")



