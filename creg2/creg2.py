#!/usr/bin/python
import sys
import json
import argparse

from sklearn import feature_extraction

from iologreg import IOLogisticRegression


parser = argparse.ArgumentParser()
parser.add_argument('label', type=str, help='labels')
parser.add_argument('training', nargs='*',
                    help='training examples (extensions ".feat" and ".resp" will be automatically appended')
parser.add_argument('--allfeatures', type=str, help='file that has all features')
parser.add_argument('--tx', type=str, help='testing features')
parser.add_argument('--ty', type=str, help='testing responses')
parser.add_argument('--output', type=str, help='output file')
parser.add_argument('--dev', action='store_true', help='tx and ty as dev set: used to tune l1')
parser.add_argument('--iterations', type=int, default=300)
parser.add_argument('--warm', type=int, default=0)
parser.add_argument('--loadmodel', type=str, help='load a trained model')
parser.add_argument('--l1', type=float, help='l1 prior (log10)')
parser.add_argument('--usingl2', action='store_true', help='use l2 instead of l1')
parser.add_argument('--bias', action='store_true', help='regularization of bias')
parser.add_argument('--honor-neighbors', action='store_true',
                    help='use neighbors specified in the .feat files (otherwise all possible labels will be used.)')
args = parser.parse_args()

features = []
labels = {}
invlabels = {}
# read labels and associated features
for line in open(args.label):
    (label, f) = line.strip().split('\t')
    invlabels[len(labels)] = label
    labels[label] = len(labels)
    features.append(json.loads(f))
label_dict = feature_extraction.DictVectorizer()
label_features = label_dict.fit_transform(features).toarray()

sys.stderr.write('        LABELS: %s\n' % ' '.join(labels.keys()))
sys.stderr.write('LABEL-FEATURES: %s\n' % ' '.join(label_dict.get_feature_names()))
out_dim = len(label_dict.get_feature_names())


def cost(y1, y2):
    s1 = set(features[y1])
    s2 = set(features[y2])
    return len(s1.intersection(s2))


def get_vectorizer(feature_file, bias={'bias': 1.0}):
    features = []
    for line in open(feature_file):
        (id, xfeats, n) = line.strip().split('\t')
        features.append(json.loads(xfeats))
    if not args.bias:
        features.append(bias)
    vectorizer = feature_extraction.DictVectorizer()
    vectorizer.fit(features)
    return vectorizer


def read_features(feature_files, response_files, vectorizer, bias={'bias': 1.0}):
    all_features = []
    all_neighbors = []
    all_responses = []

    assert len(feature_files) == len(response_files)
    for idx in range(len(feature_files)):
        ids = {}
        features = []
        neighbors = []
        # read training instances and neighborhoods
        for line in open(feature_files[idx]):
            (id, xfeats, n) = line.strip().split('\t')
            ids[id] = len(ids)
            loaded_features = json.loads(xfeats)
            if not args.bias:
                loaded_features.update(bias)
            features.append(loaded_features)
            if args.honor_neighbors:
                neighborhood = json.loads(n)['N']
                if len(neighborhood) == 0:
                    sys.stderr.write('[ERROR] empty neighborhood in line:\n%s' % line)
                    sys.exit(1)
                if len(neighborhood) == 1:
                    sys.stderr.write('[WARNING] neighborhood for id="%s" is singleton: %s\n' % (id, str(neighborhood)))
                n = [labels[x] for x in neighborhood]
            else:
                n = [x for x in labels.values()]
            neighbors.append(n)
        # read gold labels
        responses = [0 for x in xrange(len(features))]
        for line in open(response_files[idx]):
            (id, y) = line.strip().split('\t')
            responses[ids[id]] = labels[y]
        all_features.extend(features)
        all_neighbors.extend(neighbors)
        all_responses.extend(responses)
    assert len(all_features) == len(all_neighbors) == len(all_responses)
    print len(all_features)
    all_features = vectorizer.transform(all_features).toarray()
    return (all_features, all_responses, all_neighbors)


def fit_model(lbl, lbl_feat, out_dim, in_dim, X, Y, N, write_model=None, l1=1e-2, load=None, iterations=3000,
              warm_start=0, usingl2=args.usingl2, bias=None):
    def get_cost_matrix():
        costs = dict()
        for i in range(len(lbl_feat)):
            for j in range(len(lbl_feat)):
                costs[(i, j)] = cost(i, j)

        # normalize
        minimal = min(costs.values())
        maximal = max(costs.values())
        normalized = {k: 1 - (v - minimal) / (maximal - minimal) for (k, v) in costs.iteritems()}
        return normalized

    cost_matrix = get_cost_matrix()

    print 'l1: {}'.format(l1)
    assert len(X) == len(N)
    assert len(Y) == len(X)
    model = IOLogisticRegression()
    if args.bias is True:
        bias = None

    from scipy import sparse

    sparse_X = sparse.csr_matrix(X)
    import numpy as np

    # sparse_X = np.matrix(X)
    sparse_feats = np.matrix(lbl_feat)
    model.fit(in_dim, out_dim, sparse_X, N, Y, sparse_feats, len(lbl), iterations=iterations, minibatch_size=20, l1=l1,
              write=True, load_from=load, warm=warm_start, using_l2=usingl2, bias=bias, cost=cost_matrix)
    if write_model is not None:
        with open(write_model, 'w') as writer:
            writer.write(json.dumps(get_descriptive_weights(model.W, label_dict, X_dict)))
    return model


def get_descriptive_weights(W, lbl_dict, x_dict):
    import numpy as np

    weights = np.transpose(W)
    to_return = {}

    lbl_feature_list = lbl_dict.get_feature_names()
    for idx, weight_vector in enumerate(weights):
        label = lbl_feature_list[idx]
        desc_weight = x_dict.inverse_transform(weight_vector)
        to_return[label] = desc_weight
    return to_return


def predict(model, test_X, test_Y, test_N, inverse_labels, output_file='/dev/null'):
    from scipy import sparse

    sparse_X = sparse.csr_matrix(test_X)

    # sparse_X = np.matrix(test_X)
    D = model.predict_proba(sparse_X, test_N)
    to_return = []
    with open(output_file, 'w') as output_file:
        correct_count = 0
        for idx, row in enumerate(D):
            dist = {}
            for i in range(len(row)):
                if row[i] > 0.0:
                    dist[inverse_labels[i]] = row[i]
            predicted = max(dist.iterkeys(), key=lambda x: dist[x])
            answer = inverse_labels[test_Y[idx]]
            to_return.append((predicted, answer))
            output_file.write('{}\t{}\t{}\n'.format(predicted, answer, dist))
            if predicted == inverse_labels[test_Y[idx]]:
                correct_count += 1
        output_file.write('accuracy:{}\n'.format(float(correct_count) / len(D)))
    return to_return


if args.allfeatures is not None:
    X_dict = get_vectorizer(args.allfeatures)
else:
    X_dict = get_vectorizer(args.training[0] + 'feat')
in_dim = len(X_dict.get_feature_names())

sys.stderr.write('INPUT-FEATURES: %s\n' % ' '.join(X_dict.get_feature_names()))


def soft_exact(fname):
    import subprocess

    cmd = ['python', '~/git/extractor/eval.py', fname]
    results = subprocess.check_output(' '.join(cmd), shell=True)
    # ExactMatch: 0.420168067227
    # SoftMatch: 0.529579831933

    last_two_lines = results.split('\n')[-3:]
    exact = float(last_two_lines[0].strip().split()[-1])
    soft = float(last_two_lines[1].strip().split()[-1])
    return (soft, exact)


def write_csv(f_name, preds):
    with open(f_name, 'w') as outputFile:
        import csv

        writer = csv.writer(outputFile)
        writer.writerow(['predicted', 'answer', 'idx'])
        for (idx, (pred, ans)) in enumerate(preds):
            writer.writerow([pred, ans, idx])


def dev_lambda(dx_file, dy_file, X_train, Y_train, N_train):
    import numpy, math

    print dx_file
    (X_dev, Y_dev, N_dev) = read_features([dx_file], [dy_file], X_dict)

    if args.usingl2:
        r = numpy.arange(-2, 4, step=1)
    else:
        r = numpy.arange(-10, 0, step=1.7)
    for step in r:

        dev_output = 'dev_output/dev_output_{}.csv'.format(step)
        dev_output_pred = 'dev_output/dev_output_{}.pred'.format(step)

        # check if we already covered this point
        import os

        if os.path.isfile(dev_output):
            print 'lambda = {} already covered, continuing'.format(step)
            continue
        else:
            print 'lambda = {}'.format(step)

        pid = os.fork()
        if pid != 0:
            continue

        param = math.pow(10, step)
        model = fit_model(labels, label_features, out_dim, in_dim, X_train, Y_train, N_train,
                          write_model='dev_model_{}'.format(step), l1=param, iterations=args.iterations,
                          warm_start=args.warm,
                          load=args.loadmodel, bias=bias_vec)
        predictions = predict(model, X_dev, Y_dev, N_dev, invlabels, output_file=dev_output_pred)

        # softmatch and exactmatch
        # just in case


        os.system('mkdir -p dev_output')

        write_csv(dev_output, predictions)
        (soft, exact) = soft_exact(dev_output)

        with open('dev.csv', 'a') as dev_out_fh:
            import csv

            w = csv.writer(dev_out_fh)
            w.writerow([step, soft, exact])
        exit(0)
    return


training_feat = [x + 'feat' for x in args.training]
training_resp = [x + 'resp' for x in args.training]
(X, Y, N) = read_features(training_feat, training_resp, X_dict)
bias_vec = X_dict.transform([{'bias': 1.0}]).toarray()
sys.stderr.write('       rows(X): %d\n' % len(X))

if args.dev:
    dev_lambda(args.tx, args.ty, X, Y, N)
    exit()

if args.output is not None:
    output_file = args.output
else:
    output_file = 'output.pred'

if args.tx is not None and args.ty is not None:
    import numpy

    model = fit_model(labels, label_features, out_dim, in_dim, X, Y, N, 'model_output', load=args.loadmodel,
                      iterations=args.iterations, warm_start=args.warm, l1=numpy.power(10, args.l1), bias=bias_vec)

    (tX, tY, tN) = read_features([args.tx], [args.ty], X_dict)
    prediction = predict(model, tX, tY, tN, invlabels, output_file)
    write_csv(output_file + '.csv', prediction)
else:
    num_folds = 10
    from sklearn import cross_validation

    kf = cross_validation.StratifiedKFold(Y, n_folds=num_folds, indices=True)
    prediction = []
    for idx, (train, test) in enumerate(kf):
        # print train,test
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = [Y[i] for i in train], [Y[i] for i in test]
        N_train, N_test = [N[i] for i in train], [N[i] for i in test]
        model = fit_model(labels, label_features, out_dim, in_dim, X_train, Y_train, N_train, 'cv_model_{}'.format(idx),
                          bias=bias_vec)
        prediction.extend(zip(predict(model, X_test, Y_test, N_test, invlabels), test))

    with open(output_file, 'w') as outputFile:
        import csv

        writer = csv.writer(outputFile)
        writer.writerow(['predicted', 'answer', 'idx'])
        for ((pred, ans), idx) in prediction:
            writer.writerow([pred, ans, idx])
    print 'accuracy: {}/{}'.format(len([x for x in prediction if x[0][0] == x[0][1]]), len(prediction))
