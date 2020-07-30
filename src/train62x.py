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


def set_r0(src, dst):
    with open(OUT_DIR + '/' + src, 'rb') as pf:
        src_r = pickle.load(pf)

    if len(dst) != len(src_r):
        raise Exception('len(dst) != len(src_r)')

    for pos, val in enumerate(src_r):
        dst[pos][f_sample_id] = val


set_r0('_9_train.pickle', train)

n_t = np.zeros(topic_count, dtype=float)
n_u = np.zeros(user_count, dtype=np.int32)


def calculate_n():
    global n_t

    for v in train:
        u = v[f_user]

        tp = get_topic_probas(v)

        for t_, p_ in tp:
            n_t[t_] += p_

        n_u[u] += 1

    n_t = np.maximum(n_t, 1e-10)


aa_ft = np.zeros((FEATURE_COUNT, topic_count), dtype=float)
bb_fu = np.zeros((FEATURE_COUNT, user_count), dtype=float)


def set_aa_bb(feature_id, aa, bb):
    aa /= n_t
    bb /= n_u

    aa -= np.mean(aa)
    bb -= np.mean(bb)

    aa *= 1. / math.sqrt(np.mean(aa * aa))
    bb *= 1. / math.sqrt(np.mean(bb * bb))

    aa_ft[feature_id] = aa
    bb_fu[feature_id] = bb


def init_aa0_bb0():
    aa = np.zeros(topic_count, dtype=float)
    bb = np.zeros(user_count, dtype=float)

    for v in train:
        u = v[f_user]
        r0 = v[f_sample_id]

        r = v[f_target] - sigmoid(r0)

        tp = get_topic_probas(v)

        for t_, p_ in tp:
            aa[t_] += p_ * r

        bb[u] += r

    set_aa_bb(0, aa, bb)


def calculate_aa_bb(feature_id):
    prev_aa = aa_ft[feature_id - 1]
    prev_bb = bb_fu[feature_id - 1]

    aa = np.zeros(topic_count, dtype=float)
    bb = np.zeros(user_count, dtype=float)

    for v in train:
        u = v[f_user]
        r0 = v[f_sample_id]

        r = v[f_target] - sigmoid(r0)

        delta_aa = r * prev_bb[u]
        prev_aa_sum = 0.

        tp = get_topic_probas(v)

        for t_, p_ in tp:
            aa[t_] += p_ * delta_aa
            prev_aa_sum += p_ * prev_aa[t_]

        bb[u] += r * prev_aa_sum

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
        pickle.dump(aa_ft, f)

    with open(OUT_DIR + '/__' + ALG_NAME + '__bb.pickle', 'wb') as f:
        pickle.dump(bb_fu, f)


main()
