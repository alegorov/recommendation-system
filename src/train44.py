import random
import os
import pickle
from utils import *

SRC_DIR = 'data1'

ITERATION_COUNT = 497

SEED = 1

NUM_HIDDEN = 260
M_SHIFT = 1.09
LAMBDA = 0.0137
INIT_VAL = 0.01

ETA = 0.01

ALG_NAME = os.path.splitext(os.path.basename(__file__))[0].split('-')[0]
OUT_DIR = SRC_DIR + '-out'

CSV_DIR = '%s/%s-%05d' % (OUT_DIR, ALG_NAME, NUM_HIDDEN)
ALG_NAME = '%s-%05d-%05d' % (ALG_NAME, NUM_HIDDEN, SEED)

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


def set_r0(src, dst):
    with open(OUT_DIR + '/' + src, 'rb') as pf:
        src_r = pickle.load(pf)

    if len(dst) != len(src_r):
        raise Exception('len(dst) != len(src_r)')

    for pos, val in enumerate(src_r):
        dst[pos][f_sample_id] = val


set_r0('_4ic_train.pickle', train)
set_r0('_4ic_test.pickle', test)


def get_r(v, aa, bb):
    return mclip(v[f_sample_id] + np.dot(aa, bb))


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

    return get_r(v, aa, bb)


def save_result():
    with open(CSV_DIR + '/' + ALG_NAME + '.csv', 'w') as f:
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

            m = get_r(v, aa, bb) * y

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

    if not os.path.isdir(CSV_DIR):
        os.makedirs(CSV_DIR)

    save_result()

    with open(CSV_DIR + '/' + ALG_NAME + '.log', 'w') as f:
        f.write(print_str + '\n')


main()
