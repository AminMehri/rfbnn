from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np



class RBFNN:
    def __init__(self, num_centers, sigma=None):
        self.num_centers = num_centers
        self.centers = None
        self.sigma = sigma
        self.weights = None
        self.encoder = OneHotEncoder(sparse_output=False)  # Updated parameter name

    def _gaussian_rbf(self, x, center):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * self.sigma ** 2))

    def fit(self, X, y):
        # One-hot encode the labels
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1))

        # Step 1: Select the RBF centers using KMeans
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42).fit(X)
        self.centers = kmeans.cluster_centers_

        # Step 2: Calculate sigma if not provided
        if self.sigma is None:
            dists = cdist(self.centers, self.centers, 'euclidean')
            self.sigma = np.mean(dists)

        # Step 3: Compute the RBF activations for each sample
        G = np.zeros((X.shape[0], self.num_centers))
        for i, center in enumerate(self.centers):
            G[:, i] = np.exp(-np.linalg.norm(X - center, axis=1) ** 2 / (2 * self.sigma ** 2))

        # Step 4: Calculate the weights using the pseudoinverse
        self.weights = np.dot(np.linalg.pinv(G), y_encoded)

    def predict(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, center in enumerate(self.centers):
            G[:, i] = np.exp(-np.linalg.norm(X - center, axis=1) ** 2 / (2 * self.sigma ** 2))
        y_pred = np.dot(G, self.weights)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, center in enumerate(self.centers):
            G[:, i] = np.exp(-np.linalg.norm(X - center, axis=1) ** 2 / (2 * self.sigma ** 2))
        y_pred = np.dot(G, self.weights)
        return self._softmax(y_pred)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)