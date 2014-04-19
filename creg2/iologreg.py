import random
import math
from scipy.sparse import *
import sys

import numpy as np


INFINITY = float('inf')


def maximum(A, B):
    BisBigger = A - B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)


def logadd(a, b):
    """
    compute log(exp(a) + exp(b))
    """
    if a == -INFINITY:
        return b
    if b == -INFINITY:
        return a
    if b < a:  # b - a < 0
        return a + math.log1p(math.exp(b - a))
    else:  # a - b < 0
        return b + math.log1p(math.exp(a - b))


class IOLogisticRegression:
    """
    Logistic regression.
    Minimize regularized log-loss:
        L(x, y|w) = - sum_i log p(y_i|x_i, w) + l2 ||w||^2
        p(y|x, w) = exp(w[y].x) / (sum_y' exp(w[y'].x))

    Parameters
    ----------
    l2: float, default=0
        L2 regularization strength
    """

    def __init__(self, l1=1e-1, l2=1e-1):
        self.l1 = l1
        self.l2 = l2

    def gradient(self, x, n, y, y_feats, W, infeats, outfeats):
        z = -INFINITY
        log_probs = np.zeros(self.num_labels)
        xw = x.dot(W)
        found = False
        for yi in n:
            if yi == y: found = True
            # print 'x: {}, {}'.format(x, len(x))
            # print 'w: {}, {}'.format(W, len(W))
            # print 'xw: {}'.format(xw)
            # print 'lbl features: {}, {}'.format(y_feats[yi], len(y_feats[yi]))
            u = (xw * y_feats[yi].T)[0, 0]
            # print 'u: {}, {}'.format(u, len(u))
            log_probs[yi] = u
            z = logadd(z, u)
        if not found:
            print '[ERROR] for training instance', x, 'gold label', y, 'not found in neighborhood', n
            raise Exception
        loss = -(log_probs[y] - z)
        G = csr_matrix((infeats, outfeats))
        for yi in n:
            delta = math.exp(log_probs[yi] - z) - (yi == y)
            G = G + (x.T * y_feats[yi]) * delta
        return loss, G

    def fit(self, infeats, outfeats, X, N, Y, y_feats, num_labels, iterations=1000, minibatch_size=100, eta=1.0,
            l1=2., write=True, load_from=None, warm=0, using_l2=False, bias=None):
        self.l1 = l1
        print 'lambda: {}'.format(self.l1)
        minibatch_size = min(minibatch_size, X.shape[0])
        self.num_labels = num_labels
        self.y_feats = y_feats
        self.W = csr_matrix((infeats, outfeats))

        G = csr_matrix((infeats, outfeats))
        H = csr_matrix((np.ones(shape=(infeats, outfeats)) * 1e-300))
        U = csr_matrix((infeats, outfeats))
        ld = csr_matrix(np.ones(shape=(infeats, outfeats)) * self.l1)
        if bias is not None:
            if using_l2:
                bias_mask = ((np.vstack([bias] * outfeats) * (1 - self.l1))).transpose()
            else:
                bias_mask = ((np.vstack([bias] * outfeats) * (-self.l1))).transpose()

            ld = ld + bias_mask

        if load_from is not None:
            self.W = np.load(load_from)
            U = np.load(load_from[:load_from.find('.npy')] + 'U.npy')
            H = np.load(load_from[:load_from.find('.npy')] + 'H.npy')
            G = np.load(load_from[:load_from.find('.npy')] + 'G.npy')
        loss_history = []
        print '# of iterations: {}'.format(iterations)
        for i in range(warm, iterations + warm):
            sys.stderr.write('Iteration: %d\n' % i)

            loss = 0
            prior_loss = 0
            for s in random.sample(range(X.shape[0]), minibatch_size):
                tiny_loss, thisG = self.gradient(X[s], N[s], Y[s], y_feats, self.W, infeats, outfeats)
                loss += tiny_loss
                G = G + thisG
                if using_l2:
                    prior_loss += tiny_loss + self.l1 * (self.W.multiply(self.W)).sum()
                else:
                    W_copy = self.W.copy()
                    W_copy.data = np.absolute(W_copy.data)

                    prior_loss += (tiny_loss + self.l1 * W_copy.sum())

            #for k in range(self.n_classes - 1):
            #    offset = (self.n_features + 1) * k
            #    for j in range(self.n_features):
            #        loss += self.l2 * self.coef_[offset + j]**2
            #        g[offset + j] += 2 * self.l2 * self.coef_[offset + j]

            sys.stderr.write('  Loss = %f\n' % loss)
            sys.stderr.write('  Prior Loss = %f\n' % prior_loss)
            G = G / minibatch_size
            Gsqr = G.copy()
            Gsqr.data **= 2
            H = H + Gsqr
            U = U + G

            if using_l2:

                Hsqrt = - H.sqrt().multiply(ld)
                intermed = U / Hsqrt
                self.W = intermed * eta
            else:
                U_copy = U.copy()
                U_copy.data = np.absolute(U_copy.data) / (i + 1)
                U_copy = U_copy - ld
                threshold = maximum(U_copy, csr_matrix((infeats, outfeats)))

                self.W = U.sign().multiply(threshold) / H.sqrt()
                self.W = self.W * (eta * (i + 1))
                # threshold = np.maximum(np.subtract(np.divide(np.absolute(U), i + 1), ld),
                #                        np.zeros(shape=(infeats, outfeats)))
                # self.W = np.divide(np.multiply(-np.sign(U), threshold), np.sqrt(H)) * eta * (i + 1)
            if i % 50 == 0 and write is True:
                np.save('models/model_state_{}H'.format(i), H)
                np.save('models/model_state_{}'.format(i), self.W)
                np.save('models/model_state_{}U'.format(i), U)
                np.save('models/model_state_{}G'.format(i), G)
            H.eliminate_zeros()
            self.W.eliminate_zeros()
            U.eliminate_zeros()
            G.eliminate_zeros()
            print 'usage: H: {}, W: {}, U: {}, G: {}'.format(H.data.nbytes, self.W.data.nbytes, U.data.nbytes,
                                                             G.data.nbytes)
        return self

    def predict_(self, x, n, probs):
        probs.fill(0.0)
        z = -INFINITY
        xw = x.dot(self.W)
        for y in n:
            u = (xw * self.y_feats[y].T)[0, 0]
            probs[y] = u
            z = logadd(z, u)
        for y in n:
            probs[y] = math.exp(probs[y] - z)

    def predict(self, X, N):
        post = np.zeros(shape=(X.shape[0], self.num_labels))
        return post

    def predict_proba(self, X, N):
        post = np.zeros(shape=(X.shape[0], self.num_labels))
        for (x, n, p) in zip(X, N, post):
            self.predict_(x, n, p)
        return post
