import sys
from typing import Tuple, List
from math import sqrt
import random
import matplotlib.pyplot as plt # For plotting OPTIONAL, COMMENTED IN THE ORIGINAL CODE

def readfile(filename: str) -> Tuple[List, List, List]:
    headers = None
    row_names = list()
    data = list()

    with open(filename) as file_:
        for line in file_:
            values = line.strip().split("\t")
            if headers is None:
                headers = values[1:]
            else:
                row_names.append(values[0])
                data.append([float(x) for x in values[1:]])
    return row_names, headers, data


# .........DISTANCES........
# They are normalized between 0 and 1, where 1 means two vectors are identical
def euclidean(v1, v2):
    distance = sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return 1 / (1+distance)

def euclidean_squared(v1, v2):
    return euclidean(v1, v2)**2

def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
    # Sums of squares
    sum1sq = sum([v**2 for v in v1])
    sum2sq = sum([v**2 for v in v2])
    # Sum of the products
    products = sum([a * b for (a, b) in zip(v1, v2)])
    # Calculate r (Pearson score)
    num = products - (sum1 * sum2 / len(v1))
    den = sqrt((sum1sq - sum1**2 / len(v1)) * (sum2sq - sum2**2 / len(v1)))
    if den == 0:
        return 0
    return 1 - num / den


# ........HIERARCHICAL........
class BiCluster:
    def __init__(self, vec, left=None, right=None, dist=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = dist

def hcluster(rows, distance=pearson):
    distances = {}  # Cache of distance calculations
    currentclustid = -1  # Non original clusters have negative id

    # Clusters are initially just the rows
    clust = [BiCluster(row, id=i) for (i, row) in enumerate(rows)]

    """
    while ...:  # Termination criterion
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                distances[(clust[i].id, clust[j].id)] = ...

            # update closest and lowestpair if needed
            ...
        # Calculate the average vector of the two clusters
        mergevec = ...

        # Create the new cluster
        new_cluster = BiCluster(...)

        # Update the clusters
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(new_cluster)
    """

    return clust[0]

def printclust(clust: BiCluster, labels=None, n=0):
    # indent to make a hierarchy layout
    indent = " " * n
    if clust.id < 0:
        # Negative means it is a branch
        print(f"{indent}-")
    else:
        # Positive id means that it is a point in the dataset
        if labels == None:
            print(f"{indent}{clust.id}")
        else:
            print(f"{indent}{labels[clust.id]}")
    # Print the right and left branches
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n+1)


# ......... K-MEANS ..........
class KMeans:
    def __init__(self, rows, distance=euclidean_squared, k=4):
        self.rows = rows
        self.distance = distance
        self.k = k
        self.clusters = None

    def start_configuration(self, iterations=10, bestresult=None):
        for _ in range(iterations):
            initial_centroids = self.centroids_inicialization()
            centroids, totaldistance = self.assign_cluster(initial_centroids) # Centroids and total distance

            if bestresult is None or totaldistance > bestresult[1]: # If best result is None or total distance is greater than best result
                bestresult = (centroids, totaldistance) # Update best result

        return bestresult

    def centroids_inicialization(self): # Random centroids
        ranges = [(min([row[i] for row in self.rows]),
                   max([row[i] for row in self.rows])) for i in range(len(self.rows[0]))] # Get ranges for each column


        centroids = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                      for i in range(len(self.rows[0]))] for j in range(self.k)] # Get random centroids

        return centroids

    def assign_cluster(self, centroids, iterations=100, lastmatches=None):
        for _ in range(iterations):
            # print("Iteració %i" % t)
            # print(centroids[0][0])
            distances, matches = self.best_centroid(centroids)

            if matches == lastmatches: # If matches are the same as last matches stop
                break
            lastmatches = matches

            self.update_centroids(centroids, matches) # update centroids

        self.clusters = matches
        return (centroids, sum(distances))

    def best_centroid(self, centroids): # Get best centroid for each row
        bestmatches = [[] for i in range(len(centroids))]
        bestdistances = [0 for i in range(len(self.rows))]

        for idrow, row in enumerate(self.rows):
            bestmatch = 0
            bestdistance = self.distance(centroids[0], row)

            # Find the closest centroid
            for centroid_id, centroid in enumerate(centroids):
                distance = self.distance(centroid, row)
                if distance > self.distance(centroids[bestmatch], row):
                    bestmatch = centroid_id
                    bestdistance = distance

            # Store results
            bestdistances[idrow] = bestdistance
            bestmatches[bestmatch].append(idrow)

        return (bestdistances, bestmatches)

    def update_centroids(self, centroids, matches):
        # For each centroid
        for centroid, _ in enumerate(centroids):
            avgs = [0.0] * len(self.rows[0])

            if len(matches[centroid]) > 0:
                # For each item in the cluster
                for rowid in matches[centroid]:
                    # Add each value to the average
                    for attrid in range(len(self.rows[rowid])):
                        avgs[attrid] += self.rows[rowid][attrid]

                # Divide by number of items to get the average
                for j in range(len(avgs)):
                    avgs[j] /= len(matches[centroid])

                    # Update the centroid
                    centroids[centroid] = avgs

def elbow(data, begin, end, incr, restarts): # Elbow method to find the best k
    totaldistances = []
    for i in range(begin, end, incr):
        kmeans = KMeans(data, k=i)
        _, totaldistance = kmeans.start_configuration(iterations=restarts)
        totaldistances.append(totaldistance)
    return totaldistances

def plot_elbow(data, begin, end, incr, restarts): # Plot elbow method
    totaldistances = elbow(data, begin, end, incr, restarts)
    plt.plot(range(begin, end, incr), totaldistances)
    plt.show()


def main():
    try:
        filename = sys.argv[1] # Get filename from command line
    except IndexError:
        filename = "blogdata.txt" # Default filename
    _, _, data = readfile(filename)

    kmeans = KMeans(data)
    initial_centroids = kmeans.centroids_inicialization()
    """print("Initial centroids:")
    for centroid in initial_centroids:
        print(centroid)
    print()"""
    centroids, totaldistance = kmeans.assign_cluster(initial_centroids)
    """print("Final centroids: ")
    for centroid in centroids:
        print(centroid)
    print("Total distance: %f" % totaldistance)
    print()
    """

    print("Total distance: %f" % totaldistance)

    restarts = 10
    kmeans = KMeans(data)
    centroids, totaldistance = kmeans.start_configuration(iterations=restarts)
    print("%i Restarts -> Distance to centroids: %2.3f\n"
          % (restarts, totaldistance))

    result = elbow(data, 1, 10, 1, restarts)
    print("Elbow method results: " + str(result))

    for i in range(len(result)):
        print("%i Clusters - %i Restarts -> Total Distance is: %2.3f"
              % (i + 1, restarts, result[i]))

    plot_elbow(data, 1, 10, 1, restarts)

if __name__ == "__main__":
    main()
