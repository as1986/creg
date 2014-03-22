#!/usr/bin/python

__author__ = 'as1986'


def get_confu_mat(rows):
    import sys

    predictions = ([x[0] for x in rows[1:]])
    answers = ([x[1] for x in rows[1:]])
    label_set = set.union(set(predictions), set(answers))
    indexed_labels = {k: v for v, k in enumerate(label_set)}
    inversed = {v: k for v, k in enumerate(label_set)}
    one_row = [0] * len(label_set)
    table = [list(one_row) for k in range(len(label_set))]
    for (pred, ans, id) in rows[1:]:
        table[indexed_labels[pred]][indexed_labels[ans]] += 1

    for ind in range(len(label_set)):
        def f1(a, b):
            return 2 * (a * b) / (a + b)

        label = inversed[ind]
        tt = float(table[ind][ind])
        tttf = sum(table[ind]) + 1e-300
        ttft = sum([table[x][ind] for x in range(len(label_set))]) + 1e-300
        print 'label {}:'.format(label)
        print 'precision: {}'.format(tt / tttf)
        print 'recall: {}'.format(tt / ttft)
        print 'f1: {}'.format(f1(tt / tttf, tt / ttft))

    if '--nowrite' in sys.argv:
        return

    with open(sys.argv[1] + '.cm.csv', 'w') as cf:
        import csv

        writer = csv.writer(cf)
        header_row = ['X'] + [inversed[x] for x in range(len(label_set))]
        writer.writerow(header_row)
        for idx, table_row in enumerate(table):
            row = ['{}'.format(inversed[idx])] + table_row
            writer.writerow(row)

    print indexed_labels


def main():
    import csv, sys

    with open(sys.argv[1], 'r') as c:
        reader = csv.reader(c)
        rows = [x for x in reader]
        get_confu_mat(rows)


if __name__ == '__main__':
    main()
