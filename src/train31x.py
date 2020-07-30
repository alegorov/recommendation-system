import os
import pickle
from utils import *

SRC_DIR = 'data1'

FEATURE_COUNT = 100

ALG_NAME = str(os.path.splitext(os.path.basename(__file__))[0].split('-')[0])
OUT_DIR = SRC_DIR + '-out'

train = open_csv(SRC_DIR + '/train.csv')

item_count = get_item_count(train)
publisher_count = get_publisher_count(train)
user_count = get_user_count(train)
topic_count = get_topic_count(train)

print('item_count =', item_count, flush=True)
print('publisher_count =', publisher_count, flush=True)
print('user_count =', user_count, flush=True)
print('topic_count =', topic_count, flush=True)


def get_r0(t, b, c):
    return np.dot(t, b) + c


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

n_p = np.zeros(publisher_count, dtype=np.int32)
n_u = np.zeros(user_count, dtype=np.int32)


def calculate_n():
    for v in train:
        p = v[f_publisher]
        u = v[f_user]

        n_p[p] += 1
        n_u[u] += 1


aa_fp = np.zeros((FEATURE_COUNT, publisher_count), dtype=float)
bb_fu = np.zeros((FEATURE_COUNT, user_count), dtype=float)


def set_aa_bb(feature_id, aa, bb):
    aa /= n_p
    bb /= n_u

    aa -= np.mean(aa)
    bb -= np.mean(bb)

    aa *= 1. / math.sqrt(np.mean(aa * aa))
    bb *= 1. / math.sqrt(np.mean(bb * bb))

    aa_fp[feature_id] = aa
    bb_fu[feature_id] = bb


def init_aa0_bb0():
    aa = np.zeros(publisher_count, dtype=float)
    bb = np.zeros(user_count, dtype=float)

    for v in train:
        p = v[f_publisher]
        u = v[f_user]
        r0 = v[f_sample_id]

        r = v[f_target] - sigmoid(r0)

        aa[p] += r
        bb[u] += r

    set_aa_bb(0, aa, bb)


def calculate_aa_bb(feature_id):
    prev_aa = aa_fp[feature_id - 1]
    prev_bb = bb_fu[feature_id - 1]

    aa = np.zeros(publisher_count, dtype=float)
    bb = np.zeros(user_count, dtype=float)

    for v in train:
        p = v[f_publisher]
        u = v[f_user]
        r0 = v[f_sample_id]

        r = v[f_target] - sigmoid(r0)

        aa[p] += r * prev_bb[u]
        bb[u] += r * prev_aa[p]

    set_aa_bb(feature_id, aa, bb)


def main():
    calculate_n()

    for feature_id in range(FEATURE_COUNT):
        if feature_id == 0:
            init_aa0_bb0()
        else:
            calculate_aa_bb(feature_id)

        print_str = '%4d / %d' % (feature_id + 1, FEATURE_COUNT)
        print(print_str, flush=True)

    with open(OUT_DIR + '/__' + ALG_NAME + '__aa.pickle', 'wb') as f:
        pickle.dump(aa_fp, f)

    with open(OUT_DIR + '/__' + ALG_NAME + '__bb.pickle', 'wb') as f:
        pickle.dump(bb_fu, f)


main()
