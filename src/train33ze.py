import random
import os
import pickle
from utils import *

SRC_DIR = 'data1'

ITERATION_COUNT = 151

SEED = 1

E_SHIFT = 0.12
LAMBDA = 0.07
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


def get_r0(c):
    return c


def get_r(r0, aa, bb):
    return mclip(r0 + aa * bb)


with open(OUT_DIR + '/' + '_4i_c.pickle', 'rb') as pf:
    c_i = pickle.load(pf)


def calculate_r0(itr):
    for v in itr:
        i = v[f_item]
        c = c_i[i]

        v[f_sample_id] = get_r0(c)


calculate_r0(train)
calculate_r0(test)

aa_i = np.zeros(item_count, dtype=float)
bb_u = np.zeros(user_count, dtype=float)


def calculate_aa_i_bb_u():
    global aa_i
    global bb_u

    n_i = np.zeros(item_count, dtype=np.int32)
    n_u = np.zeros(user_count, dtype=np.int32)

    for v in train:
        i = v[f_item]
        u = v[f_user]
        r0 = v[f_sample_id]

        r = v[f_target] - sigmoid(r0)

        aa_i[i] += r
        n_i[i] += 1

        bb_u[u] += r
        n_u[u] += 1

    aa_i /= n_i
    bb_u /= n_u

    aa_i -= np.mean(aa_i)
    bb_u -= np.mean(bb_u)

    aa_i *= INIT_VAL / math.sqrt(np.mean(aa_i * aa_i))
    bb_u *= INIT_VAL / math.sqrt(np.mean(bb_u * bb_u))


calculate_aa_i_bb_u()


def get_r_from_v(v):
    i = v[f_item]
    u = v[f_user]
    aa = aa_i[i]
    bb = bb_u[u]

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
    global aa_i
    global bb_u

    test_has_target = f_target < len(test[0])
    best_loss = 1e10
    best_iter = 0
    print_str = ''

    for iteration in range(1, ITERATION_COUNT + 1):
        random.shuffle(train)
        loss = 0.
        for v in train:
            i = v[f_item]
            u = v[f_user]
            aa = aa_i[i]
            bb = bb_u[u]

            r0 = v[f_sample_id]

            err = v[f_target] - sigmoid(get_r(r0, aa, bb))

            loss += err * err

            if abs(err) <= E_SHIFT:
                continue

            if err > 0.:
                err -= E_SHIFT
            else:
                err += E_SHIFT

            eta = ETA * err
            eta_lambda = abs(eta) * LAMBDA

            delta_aa = eta * bb - eta_lambda * aa
            delta_bb = eta * aa - eta_lambda * bb

            aa_i[i] += delta_aa
            bb_u[u] += delta_bb

        aa_i -= np.mean(aa_i)
        bb_u -= np.mean(bb_u)

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

    save_result()

    with open(OUT_DIR + '/' + ALG_NAME + '.log', 'w') as f:
        f.write(print_str + '\n')


main()
