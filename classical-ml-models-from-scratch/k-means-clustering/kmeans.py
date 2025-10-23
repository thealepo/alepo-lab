import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def euclidean_distance(x1 , x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KMeans:
    def __init__(self , K=5 , max_iterations=100 , plot_steps=False):
        self.K = K
        self.max_iterations = max_iterations
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # store mean feature vector per cluster
        self.centroids = []

    def predict(self , X):
        self.X = X
        self.n_samples , self.n_features = X.shape

        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples , self.K , replace=False)  # array of size self.K
        self.centroids = [self.X[i] for i in random_sample_idxs]

        # optimize
        for _ in range(self.max_iterations):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()
            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self.plot_steps:
                self.plot()
            # check convergence
            if self._is_converged(centroids_old , self.centroids):
                break

        # return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self , centroids):
        clusters = [[] for _ in range(self.K)]
        
        for i,sample in enumerate(self.X):
            centroid_i = self._closest_centroid(sample , centroids)
            clusters[centroid_i].append(i)

        return clusters

    def _closest_centroid(self , sample , centroids):
        distances = [euclidean_distance(sample , point) for point in centroids]
        return np.argmin(distances)  # returning closest index

    def _get_centroids(self , clusters):
        centroids = np.zeros((self.K , self.n_features))  # tuples

        for i,cluster in enumerate(clusters):
            if not cluster:
                continue

            cluster_mean = np.mean(self.X[cluster] , axis=0)
            centroids[i] = cluster_mean

        return centroids

    def _is_converged(self , old , new):
        distances = [euclidean_distance(old[i] , new[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_cluster_labels(self , clusters):
        labels = np.empty(self.n_samples)  # index of cluster it was assigned to

        for i,cluster in enumerate(clusters):
            for j in cluster:
                labels[j] = i
        
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        labels = self._get_cluster_labels(self.clusters)

        ax.scatter(self.X[:, 0] , self.X[:, 1] , c=labels , cmap='viridis' , s=50 , alpha=0.8)

        for point in self.centroids:
            ax.scatter(*point , marker='x' , color='black' , linewidth=3 , s=150)
            
        plt.show()
