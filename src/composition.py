from utils import *
from rprop import *
import glob
import os

DIR0 = 'data0'
DIR1 = 'data1'

ITERATION_COUNT = 10000
INITIAL_STEP = 0.001


def get_test_y():
    data = open_csv(DIR1 + '/test.csv')
    data = list(map(lambda v: (1. if v[f_target] else -1.), data))
    return data


def load_input_data(data_dir):
    names = glob.glob(data_dir + '/*.csv')
    names = list(map(lambda s: os.path.splitext(os.path.basename(s))[0], names))
    names.sort()

    names.insert(0, '1')

    data = [[]] * len(names)
    for z in range(1, len(names)):
        a = open_pred_csv(data_dir + '/' + names[z] + '.csv')
        data[z] = list(map(lambda v: sigmoid1(v[1]), a))

    data[0] = [1.] * len(data[1])

    data = list(zip(*data))
    data = np.array(data, dtype=float)
    return data, names


def print_accuracy(iteration, loss):
    log = "{:10d}: score = {:.13f}"
    log_str = log.format(iteration - 1, score(loss))
    print(log_str, flush=True)


def main():
    test_y = get_test_y()
    data0, names0 = load_input_data(DIR0 + '-out')
    data1, names1 = load_input_data(DIR1 + '-out')

    if names0 != names1:
        raise Exception('names0 != names1')

    w = [1 / (len(names1) - 1)] * len(names1)
    w[0] = 0.
    w = list(map(lambda x: [x, 0.], w))

    for iteration in range(1, ITERATION_COUNT + 1):
        grad = np.zeros(len(names1), dtype=float)
        w0 = np.array(list(map(lambda x: x[0], w)), dtype=float)

        loss = 0.

        for i, y in enumerate(test_y):
            m = mclip(np.dot(data1[i], w0)) * y

            loss -= log_sigmoid(m)

            eta = sigmoid(-m) * y

            grad += eta * data1[i]

        loss /= len(test_y)

        print_accuracy(iteration, loss)

        for z in range(len(names1)):
            rprop(w[z], grad[z], initial_stepsize=INITIAL_STEP)

    w = np.array(list(map(lambda x: x[0], w)), dtype=float)

    test = open_csv(DIR0 + '/test.csv')

    with open('pred.csv', 'w') as f:
        f.write('sample_id,target\n')
        for i, v in enumerate(test):
            r = np.dot(data0[i], w)
            r = sigmoid(mclip(r))
            f.write('%s,%s\n' % (v[f_sample_id], r))

    print('\n**************************  w ***************************')

    zs = list(range(len(names1)))
    zs.sort(key=lambda zz: -abs(w[zz]))

    for i, z in enumerate(zs, 1):
        print('%5d %30s: %s' % (i, names1[z], w[z]))


main()
