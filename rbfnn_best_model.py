import numpy as np

def gaussian_rbf(x, c, sigma):
    return np.exp(-np.linalg.norm(x - c)**2 / (2 * sigma**2))

class RBFNetwork:
    def __init__(self, centers, sigma=1.0, l2_reg=0.0):
        self.centers = centers
        self.sigma = sigma
        self.l2_reg = l2_reg
        self.weights = None

    def _design_matrix(self, X):
        return np.array([[gaussian_rbf(x, c, self.sigma) for c in self.centers] for x in X])

    def fit(self, X, y):
        phi = self._design_matrix(X)
        if self.l2_reg > 0:
            # اضافه کردن تنظیم L2 (ریج)
            reg_matrix = self.l2_reg * np.eye(phi.shape[1])
            self.weights = np.linalg.lstsq(phi.T @ phi + reg_matrix, phi.T @ y, rcond=None)[0]
        else:
            self.weights = np.linalg.lstsq(phi.T @ phi, phi.T @ y, rcond=None)[0]

    def predict(self, X):
        phi = self._design_matrix(X)
        return phi @ self.weights
