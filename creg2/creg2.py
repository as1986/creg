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
parser.add_argument('--dev', action='store_true', help='')
parser.add_argument('--iterations', type=int, default=300)
parser.add_argument('--warm', type=int, default=0)
parser.add_argument('--loadmodel', type=str, help='load a trained model')
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


def get_vectorizer(feature_file):
    features = []
    for line in open(feature_file):
        (id, xfeats, n) = line.strip().split('\t')
        features.append(json.loads(xfeats))
    vectorizer = feature_extraction.DictVectorizer()
    vectorizer.fit(features)
    return vectorizer


def read_features(feature_files, response_files, vectorizer):
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
            features.append(json.loads(xfeats))
            neighborhood = json.loads(n)['N']
            if len(neighborhood) == 0:
                sys.stderr.write('[ERROR] empty neighborhood in line:\n%s' % line)
                sys.exit(1)
            if len(neighborhood) == 1:
                sys.stderr.write('[WARNING] neighborhood for id="%s" is singleton: %s\n' % (id, str(neighborhood)))
            n = [labels[x] for x in neighborhood]
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


def fit_model(lbl, lbl_feat, out_dim, in_dim, X, Y, N, write_model=None, l1=1e-2, load=None, iterations=3000, warm_start=0):
    assert len(X) == len(N)
    assert len(Y) == len(X)
    model = IOLogisticRegression()
    model.fit(in_dim, out_dim, X, N, Y, lbl_feat, len(lbl), iterations=iterations, minibatch_size=20, l1=l1, write=True, load_from=load, warm=warm_start)
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
    D = model.predict_proba(test_X, test_N)
    to_return = []
    with open(output_file, 'w') as outputFile:
        correct_count = 0
        for idx, row in enumerate(D):
            dist = {}
            for i in range(len(row)):
                if row[i] > 0.0:
                    dist[inverse_labels[i]] = row[i]
            predicted = max(dist.iterkeys(), key=lambda x: dist[x])
            answer = inverse_labels[test_Y[idx]]
            to_return.append((predicted, answer))
            outputFile.write('{}\t{}\t{}\n'.format(predicted, answer, dist))
            if predicted == inverse_labels[test_Y[idx]]:
                correct_count += 1
        outputFile.write('accuracy:{}\n'.format(float(correct_count) / len(D)))
    return to_return


if args.allfeatures is not None:
    X_dict = get_vectorizer(args.allfeatures)
else:
    X_dict = get_vectorizer(args.training[0] + 'feat')
in_dim = len(X_dict.get_feature_names())

sys.stderr.write('INPUT-FEATURES: %s\n' % ' '.join(X_dict.get_feature_names()))


def dev_lambda(dx_file, dy_file, X_train, Y_train, N_train):
    print dx_file
    which_dev = []
    (X_dev, Y_dev, N_dev) = read_features([dx_file], [dy_file], X_dict)
    for step in range(-5,0):
        import numpy
        param = numpy.power(10, step)
        model = fit_model(labels, label_features, out_dim, in_dim, X_train, Y_train, N_train, 'dev_model_{}'.format(step), l1=param)
        predictions = predict(model, X_dev, Y_dev, N_dev, invlabels)
        which_dev.append((step, len([x for x in predictions if x[0] == x[1]])))
        print which_dev[-1]
    return which_dev


training_feat = [x + 'feat' for x in args.training]
training_resp = [x + 'resp' for x in args.training]
(X, Y, N) = read_features(training_feat, training_resp, X_dict)
sys.stderr.write('       rows(X): %d\n' % len(X))


if args.dev:
    print dev_lambda(args.tx, args.ty, X, Y, N)
    exit()

if args.output is not None:
    output_file = args.output
else:
    output_file = 'output.pred'

if args.tx is not None and args.ty is not None:
    model = fit_model(labels, label_features, out_dim, in_dim, X, Y, N, 'model_output', load=args.loadmodel, iterations=args.iterations, warm_start=args.warm)

    (tX, tY, tN) = read_features([args.tx], [args.ty], X_dict)
    predict(model, tX, tY, tN, invlabels, output_file)
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
        model = fit_model(labels, label_features, out_dim, in_dim, X_train, Y_train, N_train, 'cv_model_{}'.format(idx))
        prediction.extend(zip(predict(model, X_test, Y_test, N_test, invlabels), test))

    with open(output_file, 'w') as outputFile:
        import csv

        writer = csv.writer(outputFile)
        writer.writerow(['predicted', 'answer', 'idx'])
        for ((pred, ans), idx) in prediction:
            writer.writerow([pred, ans, idx])
    print 'accuracy: {}/{}'.format(len([x for x in prediction if x[0][0] == x[0][1]]), len(prediction))
