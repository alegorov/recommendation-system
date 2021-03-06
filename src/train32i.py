import random
import os
import pickle
from utils import *

SRC_DIR = 'data1'

ITERATION_COUNT = 472

SEED = 1

NUM_HIDDEN = 260
M_SHIFT = 1.2
LAMBDA = 0.02
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


def get_r0(t, b, c):
    return np.dot(t, b) + c


def get_r(r0, aa, bb):
    return mclip(r0 + np.dot(aa, bb))


with open(OUT_DIR + '/' + '_8_b.pickle', 'rb') as pf:
    b_ut = pickle.load(pf)

with open(OUT_DIR + '/' + '_8_c.pickle', 'rb') as pf:
    c_i = pickle.load(pf)


def calculate_r0(itr):
    for v in itr:
        i = v[f_item]
        u = v[f_user]
        b = b_ut[u]
        c = c_i[i]

        t = get_topic_vect(v, topic_count)

        v[f_sample_id] = get_r0(t, b, c)


calculate_r0(train)
calculate_r0(test)

aa_tf = np.random.uniform(-INIT_VAL, INIT_VAL, (topic_count, NUM_HIDDEN))
bb_uf = np.zeros((user_count, NUM_HIDDEN), dtype=float)

n_u = np.zeros(user_count, dtype=np.int32)
norm_coef = 1.


def calculate_n_u():
    for v in train:
        u = v[f_user]
        n_u[u] += 1


calculate_n_u()


def calculate_bb_uf():
    global bb_uf
    global norm_coef

    bb_uf.fill(0.)

    for v in train:
        u = v[f_user]
        r0 = v[f_sample_id]

        err0 = v[f_target] - sigmoid(r0)

        tp = get_topic_probas(v)

        for t_, p_ in tp:
            bb_uf[u] += (p_ * err0) * aa_tf[t_]

    bb_uf /= n_u[:, np.newaxis]
    bb_uf -= np.mean(bb_uf, axis=0)

    norm_coef = math.sqrt(np.mean(aa_tf * aa_tf) / np.mean(bb_uf * bb_uf))
    bb_uf *= norm_coef


aa_tf -= np.mean(aa_tf, axis=0)
calculate_bb_uf()


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

            r0 = v[f_sample_id]

            y = 1. if v[f_target] else -1.

            m = get_r(r0, aa, bb) * y

            loss -= log_sigmoid(m)

            eta = ETA * sigmoid(-m - M_SHIFT) * y

            dc = (v[f_target] - sigmoid(r0)) / n_u[u]
            dc *= norm_coef

            coef2 = eta * dc - ETA * LAMBDA

            for t_, p_ in tp:
                aa_tf[t_] += (p_ * eta) * bb + (p_ * coef2) * aa_tf[t_]

        aa_tf -= np.mean(aa_tf, axis=0)
        calculate_bb_uf()

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
