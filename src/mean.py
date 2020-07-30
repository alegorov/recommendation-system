from utils import *
import glob
import os
import sys

DIR0 = 'data0'
DIR1 = 'data1'

ALG_NAME = sys.argv[1]


def get_test_y():
    data = open_csv(DIR1 + '/test.csv')
    data = list(map(lambda v: (1. if v[f_target] else -1.), data))
    return data


def load_input_data(data_dir):
    names = glob.glob(data_dir + '/*.csv')
    names = list(map(lambda s: os.path.splitext(os.path.basename(s))[0], names))
    names.sort()
    data = [[]] * len(names)
    ids0 = None

    for z in range(len(names)):
        a = open_pred_csv(data_dir + '/' + names[z] + '.csv')
        ids = list(map(lambda v: v[0], a))
        if z:
            if ids != ids0:
                raise Exception('ids != ids0')
        else:
            ids0 = ids

        data[z] = list(map(lambda v: sigmoid1(v[1]), a))

    data = list(zip(*data))
    data = np.array(data, dtype=float)
    data = np.mean(data, axis=1)
    return ids0, data, names


def main():
    test_y = get_test_y()
    ids0, data0, names0 = load_input_data(DIR0 + '-out/' + ALG_NAME)
    ids1, data1, names1 = load_input_data(DIR1 + '-out/' + ALG_NAME)

    if names0 != names1:
        raise Exception('names0 != names1')

    loss = 0.

    with open(DIR1 + '-out/' + ALG_NAME + '.csv', 'w') as f:
        f.write('sample_id,target\n')

        for i, y in enumerate(test_y):
            c = mclip(data1[i])
            r = sigmoid(c)
            f.write('%s,%s\n' % (ids1[i], r))
            m = c * y

            loss -= log_sigmoid(m)

    loss /= len(test_y)

    log_str = '\n'.join(names0) + '\n'
    log_str += '------------------------------------------\n'

    log = "loss = {:.6f}, score = {:.13f}"
    log_str += log.format(loss, score(loss))
    log_str += '\n------------------------------------------'
    print(log_str, flush=True)

    test = open_csv(DIR0 + '/test.csv')

    with open(DIR0 + '-out/' + ALG_NAME + '.csv', 'w') as f:
        f.write('sample_id,target\n')
        for i, v in enumerate(test):
            if ids0[i] != v[f_sample_id]:
                raise Exception('ids0[i] != v[f_sample_id]')
            r = sigmoid(mclip(data0[i]))
            f.write('%s,%s\n' % (ids0[i], r))

    with open(DIR0 + '-out/' + ALG_NAME + '.log', 'w') as f:
        f.write(log_str + '\n')

    with open(DIR1 + '-out/' + ALG_NAME + '.log', 'w') as f:
        f.write(log_str + '\n')


main()
