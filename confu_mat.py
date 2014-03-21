#!/usr/bin/python

__author__ = 'as1986'


def get_confu_mat(rows):
    predictions = ([x[0] for x in rows[1:]])
    answers = ([x[1] for x in rows[1:]])
    label_set = set.union(set(predictions), set(answers))
    indexed_labels = {k: v for v, k in enumerate(label_set)}
    one_row = [1] * len(label_set)
    table = one_row * len(label_set)
    for (pred, ans, id) in rows[1:]:
        table[indexed_labels[pred]][indexed_labels[ans]] += 1

    print ' \t' + ' '.join([str(x) for x in range(len(label_set))])
    for idx, table_row in enumerate(table):
        print '{}\t'.format(idx) + ' '.join(table_row)

    print indexed_labels


def main():
    import csv, sys

    with open(sys.argv[1], 'r') as c:
        reader = csv.reader(c)
        rows = [x for x in reader]
        get_confu_mat(rows)


if __name__ == '__main__':
    main()