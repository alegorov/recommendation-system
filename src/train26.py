import random
import os
import pickle
from utils import *

SRC_DIR = 'data1'

ITERATION_COUNT = 475

SEED = 1

NUM_HIDDEN = 260
M_SHIFT = 1.62
LAMBDA = 0.0133
INIT_VAL = 0.01

ETA = 0.03

ALG_NAME = str(os.path.splitext(os.path.basename(__file__))[0].split('-')[0])
OUT_DIR = SRC_DIR + '-out'

random.seed(SEED)
random.seed(random.randint(1, 1000000000))
np.random.seed(random.randint(1, 1000000000))

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

test_ids = list(map(lambda v: v[f_sample_id], test))


def get_r0(a, t, b):
    return a + np.dot(t, b)


def get_r(r0, aa, bb):
    return mclip(r0 + np.dot(aa, bb))


with open(OUT_DIR + '/' + '_0_a.pickle', 'rb') as pf:
    a_p = pickle.load(pf)

with open(OUT_DIR + '/' + '_0_b.pickle', 'rb') as pf:
    b_ut = pickle.load(pf)


def calculate_r0(itr):
    for v in itr:
        p = v[f_publisher]
        u = v[f_user]
        a = a_p[p]
        b = b_ut[u]

        t = get_topic_vect(v, topic_count)

        v[f_sample_id] = get_r0(a, t, b)


calculate_r0(train)
calculate_r0(test)

train_org = train.copy()

aa_tf = np.random.uniform(-INIT_VAL, INIT_VAL, (topic_count, NUM_HIDDEN))
bb_uf = np.random.uniform(-INIT_VAL, INIT_VAL, (user_count, NUM_HIDDEN))

aa_tf -= np.mean(aa_tf, axis=0)
bb_uf -= np.mean(bb_uf, axis=0)


def get_r_from_v(v):
    u = v[f_user]

    tp = get_topic_probas(v)

    aa = np.zeros(NUM_HIDDEN, dtype=float)

    for t_, p_ in tp:
        aa += p_ * aa_tf[t_]

    bb = bb_uf[u]

    r0 = v[f_sample_id]

    return get_r(r0, aa, bb)


def save_result():
    with open(OUT_DIR + '/' + ALG_NAME + '.csv', 'w') as f:
        f.write('sample_id,target\n')
        for i, v in enumerate(test):
            r = max(min(sigmoid(get_r_from_v(v)), 1.), 0.)
            f.write('%s,%s\n' % (test_ids[i], r))


def get_test_loss():
    loss = 0.

    for v in test:
        y = 1. if v[f_target] else -1.

        m = get_r_from_v(v) * y

        loss -= log_sigmoid(m)

    return loss / len(test)


def main():
    global aa_tf
    global bb_uf

    test_has_target = f_target < len(test[0])
    best_loss = 1e10
    best_iter = 0
    print_str = ''

    for iteration in range(1, ITERATION_COUNT + 1):
        random.shuffle(train)
        loss = 0.
        for v in train:
            u = v[f_user]

            tp = get_topic_probas(v)

            aa = np.zeros(NUM_HIDDEN, dtype=float)

            for t_, p_ in tp:
                aa += p_ * aa_tf[t_]

            bb = bb_uf[u]

            y = 1. if v[f_target] else -1.

            r0 = v[f_sample_id]

            m = get_r(r0, aa, bb) * y

            loss -= log_sigmoid(m)

            eta = ETA * sigmoid(-m - M_SHIFT) * y

            delta_bb = eta * aa - (ETA * LAMBDA) * bb

            for t_, p_ in tp:
                delta_aa = (eta * p_) * bb - (ETA * LAMBDA * p_) * aa_tf[t_]
                aa_tf[t_] += delta_aa

            bb_uf[u] += delta_bb

        aa_tf -= np.mean(aa_tf, axis=0)
        bb_uf -= np.mean(bb_uf, axis=0)

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

    train_r = [get_r_from_v(v) for v in train_org]
    test_r = [get_r_from_v(v) for v in test]

    with open(OUT_DIR + '/' + '_26_train.pickle', 'wb') as f:
        pickle.dump(train_r, f)

    with open(OUT_DIR + '/' + '_26_test.pickle', 'wb') as f:
        pickle.dump(test_r, f)

    save_result()

    with open(OUT_DIR + '/' + ALG_NAME + '.log', 'w') as f:
        f.write(print_str + '\n')


main()
