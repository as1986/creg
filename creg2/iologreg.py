import numpy as np
import random
import math
import sys

INFINITY = float('inf')


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

    def gradient(self, x, n, y, y_feats, W, G):
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
            u = xw.dot(y_feats[yi])
            # print 'u: {}, {}'.format(u, len(u))
            log_probs[yi] = u
            z = logadd(z, u)
        if not found:
            print '[ERROR] for training instance', x, 'gold label', y, 'not found in neighborhood', n
            raise Exception
        loss = -(log_probs[y] - z)
        for yi in n:
            delta = math.exp(log_probs[yi] - z) - (yi == y)
            G += np.outer(x, y_feats[yi]) * delta
        return loss

    def fit(self, infeats, outfeats, X, N, Y, y_feats, num_labels, iterations=1000, minibatch_size=100, eta=1.0,
            l1=2., write=True, load_from=None, warm=0, using_l2=False, bias=None):
        self.l1 = l1
        minibatch_size = min(minibatch_size, len(X))
        self.num_labels = num_labels
        self.y_feats = y_feats
        self.W = np.zeros(shape=(infeats, outfeats))

        G = np.zeros(shape=(infeats, outfeats))
        H = np.ones(shape=(infeats, outfeats)) * 1e-300
        U = np.zeros(shape=(infeats, outfeats))
        ld = np.ones(shape=(infeats, outfeats)) * self.l1
        if using_l2: 
            bias_mask = np.array((np.vstack([bias] * len(y_feats)) * (1-self.l1))).transpose()
        else:
            bias_mask = np.array((np.vstack([bias] * len(y_feats)) * (-self.l1))).transpose()
            
        ld += bias_mask

        if load_from is not None:
            self.W = np.load(load_from)
            U = np.load(load_from[:load_from.find('.npy')] + 'U.npy')
            H = np.load(load_from[:load_from.find('.npy')] + 'H.npy')
            G = np.load(load_from[:load_from.find('.npy')] + 'G.npy')
        loss_history = []
        for i in range(warm, iterations + warm):
            sys.stderr.write('Iteration: %d\n' % i)
            G.fill(0.0)
            loss = 0
            prior_loss = 0
            for s in random.sample(range(X.shape[0]), minibatch_size):
                tiny_loss = self.gradient(X[s], N[s], Y[s], y_feats, self.W, G)
                loss += tiny_loss
                if using_l2:
                    prior_loss += tiny_loss + self.l1 * np.sum(np.multiply(self.W, self.W))
                else:
                    prior_loss += (tiny_loss + self.l1 * np.sum(np.absolute(self.W)))

            #for k in range(self.n_classes - 1):
            #    offset = (self.n_features + 1) * k
            #    for j in range(self.n_features):
            #        loss += self.l2 * self.coef_[offset + j]**2
            #        g[offset + j] += 2 * self.l2 * self.coef_[offset + j]

            sys.stderr.write('  Loss = %f\n' % loss)
            sys.stderr.write('  Prior Loss = %f\n' % prior_loss)
            G /= minibatch_size
            H += np.square(G)
            U += G

            if using_l2:
                intermed = np.divide(-U, np.multiply(np.sqrt(H), ld))
                self.W = intermed * eta
            else:
                threshold = np.maximum(np.subtract(np.divide(np.absolute(U), i + 1), ld),
                                       np.zeros(shape=(infeats, outfeats)))
                self.W = np.divide(np.multiply(-np.sign(U), threshold), np.sqrt(H)) * eta * (i + 1)
            if i % 50 == 0 and write is True:
                np.save('models/model_state_{}H'.format(i), H)
                np.save('models/model_state_{}'.format(i), self.W)
                np.save('models/model_state_{}U'.format(i), U)
                np.save('models/model_state_{}G'.format(i), G)
        return self

    def predict_(self, x, n, probs):
        probs.fill(0.0)
        z = -INFINITY
        xw = x.dot(self.W)
        for y in n:
            u = xw.dot(self.y_feats[y])
            probs[y] = u
            z = logadd(z, u)
        for y in n:
            probs[y] = math.exp(probs[y] - z)

    def predict(self, X, N):
        post = np.zeros(shape=(len(X), self.num_labels))
        return post

    def predict_proba(self, X, N):
        post = np.zeros(shape=(len(X), self.num_labels))
        for (x, n, p) in zip(X, N, post):
            self.predict_(x, n, p)
        return post
