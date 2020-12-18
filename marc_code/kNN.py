import numpy as np
import utils


class kNNAlgorithm:

    def __init__(self, k, r, X_train, X_test, y_train, edited=0):
        self.k = k
        self.r = r
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.edited = edited

    def get_distances(self):
        return utils.Minkowski(r=self.r, x=self.X_train, y=self.X_test)

    def get_kNN_index(self):
        d = self.get_distances()

        if self.edited == 1:
            d[np.eye(d.shape[0], dtype=bool)] = 1000

        idx = np.argsort(d.T)
        final_idx = idx[:, :self.k]

        return final_idx, d

    def max_voting(self):
        kNN_index = self.get_kNN_index()[0]
        kNN_labels = self.y_train[kNN_index]

        y_pred = utils.mode(kNN_labels).reshape(self.X_test.shape[0])

        return y_pred

    def inverse_distance_voting(self):
        kNN_index, d = self.get_kNN_index()
        kNN_labels = self.y_train[kNN_index]

        neigh_dist = np.zeros_like(kNN_index, dtype=float)
        for i in range(d.shape[1]):
            neigh_dist[i, :] = d.T[i, kNN_index[i, :]]

        vote = np.zeros((neigh_dist.shape[0], max(self.y_train)+1))
        for cls in np.unique(self.y_train):
            for j in range(neigh_dist.shape[0]):
                for k in range(self.k):
                    if kNN_labels[j, k] == cls:
                        vote[j, cls] = vote[j, cls] + 1/neigh_dist[j, k]
                    else:
                        pass

        y_pred = np.argmax(vote, axis=1)

        return y_pred

    def sheppard_voting(self):
        kNN_index, d = self.get_kNN_index()
        kNN_labels = self.y_train[kNN_index]

        neigh_dist = np.zeros_like(kNN_index, dtype=float)
        for i in range(d.shape[1]):
            neigh_dist[i, :] = d.T[i, kNN_index[i, :]]

        vote = np.zeros((neigh_dist.shape[0], max(self.y_train) + 1))
        for cls in np.unique(self.y_train):
            for j in range(neigh_dist.shape[0]):
                for k in range(self.k):
                    if kNN_labels[j, k] == cls:
                        vote[j, cls] = vote[j, cls] + np.exp(-neigh_dist[j, k])
                    else:
                        pass

        y_pred = np.argmax(vote, axis=1)

        return y_pred

    def stochastic_edited_pr(self):
        kNN_index, d = self.get_kNN_index()
        kNN_labels = self.y_train[kNN_index]

        neigh_dist = np.zeros_like(kNN_index, dtype=float)
        for i in range(d.shape[1]):
            neigh_dist[i, :] = d.T[i, kNN_index[i, :]]

        pr = np.zeros((neigh_dist.shape[0], max(self.y_train) + 1))
        for cls in np.unique(self.y_train):
            for j in range(neigh_dist.shape[0]):
                for k in range(self.k):
                    if kNN_labels[j, k] == cls:
                        pr[j, cls] = pr[j, cls] + 1 / (1 + neigh_dist[j, k])
                    else:
                        pass

        # Normalize pr
        total_pr = np.sum(pr, axis=1)[:, np.newaxis]
        pr = pr/total_pr

        # Obtain max pr and its associated label
        y_pred = np.argmax(pr, axis=1)
        max_pr = pr.max(axis=1)

        return y_pred, max_pr



