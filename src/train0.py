import random
import os
import pickle
from utils import *

SRC_DIR = 'data0'
ITERATION_COUNT = 329

ETA = 0.002

ALG_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = SRC_DIR + '-out'

random.seed(83659)

train = open_csv(SRC_DIR + '/train.csv')
test = open_csv(SRC_DIR + '/test.csv')

item_count = get_item_count(train)
publisher_count = get_publisher_count(train)
user_count = get_user_count(train)
topic_count = get_topic_count(train)

if item_count != get_item_count(test):
    raise Exception('item_count != get_item_count(test)')
if publisher_count != get_publisher_count(test):
    raise Exception('publisher_count != get_publisher_count(test)')
if user_count != get_user_count(test):
    raise Exception('user_count != get_user_count(test)')
if topic_count != get_topic_count(test):
    raise Exception('topic_count != get_topic_count(test)')

print('item_count =', item_count, flush=True)
print('publisher_count =', publisher_count, flush=True)
print('user_count =', user_count, flush=True)
print('topic_count =', topic_count, flush=True)


def get_r(a, t, b):
    return a + np.dot(t, b)


a_p = np.zeros(publisher_count, dtype=float)
b_ut = np.zeros((user_count, topic_count), dtype=float)


def get_r_from_v(v):
    u = v[f_user]
    p = v[f_publisher]
    a = a_p[p]
    b = b_ut[u]

    t = get_topic_vect(v, topic_count)

    return get_r(a, t, b)


def save_result():
    with open(OUT_DIR + '/' + ALG_NAME + '.csv', 'w') as f:
        f.write('sample_id,target\n')
        for v in test:
            r = max(min(sigmoid(get_r_from_v(v)), 1.), 0.)
            f.write('%s,%s\n' % (v[f_sample_id], r))


def get_test_loss():
    loss = 0.

    for v in test:
        y = 1. if v[f_target] else -1.

        m = get_r_from_v(v) * y

        loss -= log_sigmoid(m)

    return loss / len(test)


def main():
    test_has_target = f_target < len(test[0])
    best_loss = 1e10
    best_iter = 0
    print_str = ''

    for iteration in range(1, ITERATION_COUNT + 1):
        random.shuffle(train)
        loss = 0.
        for v in train:
            u = v[f_user]
            p = v[f_publisher]
            a = a_p[p]
            b = b_ut[u]

            y = 1. if v[f_target] else -1.

            t = get_topic_vect(v, topic_count)

            m = get_r(a, t, b) * y

            loss -= log_sigmoid(m)

            eta = ETA * sigmoid(-m) * y

            a_p[p] += eta
            b_ut[u] += eta * t

        loss /= len(train)

        if test_has_target:
            test_loss = get_test_loss()
            if test_loss < best_loss:
                best_loss = test_loss
                best_iter = iteration
            print_str = '%4d: train = %.6f, test = %.6f, best = %.6f (%s)' % (
                iteration, loss, test_loss, best_loss, best_iter)
        else:
            print_str = '%4d: train = %.6f' % (iteration, loss)

        print(print_str, flush=True)

    with open(OUT_DIR + '/' + '_0_a.pickle', 'wb') as f:
        pickle.dump(a_p, f)

    with open(OUT_DIR + '/' + '_0_b.pickle', 'wb') as f:
        pickle.dump(b_ut, f)

    # save_result()

    with open(OUT_DIR + '/' + ALG_NAME + '.log', 'w') as f:
        f.write(print_str + '\n')


main()
