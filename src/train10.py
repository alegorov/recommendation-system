import os
import pickle
from utils import *
from catboost import CatBoostClassifier, Pool

SRC_DIR = 'data0'
ITERATION_COUNT = 1225

TREE_DEPTH = 1
BORDER_COUNT = 256
RANDOM_STRENGTH = 1060.

ETA = 0.01

ALG_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = SRC_DIR + '-out'

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


def v2data_(v, a_p, b_ut, b0_ut, b1_ut, c_i, d_u):
    i = v[f_item]
    u = v[f_user]
    p = v[f_publisher]

    t = get_topic_vect(v, topic_count)

    return [
        a_p[p],
        np.dot(t, b_ut[u]),
        np.dot(t, b0_ut[u]),
        np.dot(t, b1_ut[u]),
        c_i[i],
        d_u[u],
        a_p[p] + c_i[i],
        a_p[p] + d_u[u],
        c_i[i] + d_u[u],
        a_p[p] + c_i[i] + d_u[u]
    ]


def csv2data_():
    n_p = np.zeros(publisher_count, dtype=float)
    n_i = np.zeros(item_count, dtype=float)
    n_u = np.zeros(user_count, dtype=float)

    a_p = np.zeros(publisher_count, dtype=float)
    b_ut = np.zeros((user_count, topic_count), dtype=float)
    b0_ut = np.zeros((user_count, topic_count), dtype=float)
    b1_ut = np.zeros((user_count, topic_count), dtype=float)
    c_i = np.zeros(item_count, dtype=float)
    d_u = np.zeros(user_count, dtype=float)

    for v in train:
        i = v[f_item]
        u = v[f_user]
        p = v[f_publisher]

        n_p[p] += 1.
        n_i[i] += 1.
        n_u[u] += 1.

        y = 1. if v[f_target] else -1.

        t = get_topic_vect(v, topic_count)

        a_p[p] += y

        if v[f_target]:
            b_ut[u] += t
            b1_ut[u] += t
        else:
            b_ut[u] -= t
            b0_ut[u] += t

        c_i[i] += y
        d_u[u] += y

    a_p /= n_p
    b_ut /= n_u[:, np.newaxis]
    b0_ut /= n_u[:, np.newaxis]
    b1_ut /= n_u[:, np.newaxis]
    c_i /= n_i
    d_u /= n_u

    train_data = [[]] * len(train)
    test_data = [[]] * len(test)

    for pos, v in enumerate(train):
        train_data[pos] = v2data_(v, a_p, b_ut, b0_ut, b1_ut, c_i, d_u)

    for pos, v in enumerate(test):
        test_data[pos] = v2data_(v, a_p, b_ut, b0_ut, b1_ut, c_i, d_u)

    train_data = np.array(train_data, dtype=float)
    test_data = np.array(test_data, dtype=float)

    return train_data, test_data


# def v2data1(v, a_p, b_ut, c_i, d_u):
#     i = v[f_item]
#     u = v[f_user]
#     p = v[f_publisher]
#
#     t = get_topic_vect(v, topic_count)
#
#     tb = np.dot(t, b_ut[u])
#
#     return [
#         a_p[p],
#         tb,
#         c_i[i],
#         d_u[u],
#         a_p[p] + c_i[i],
#         a_p[p] + d_u[u],
#         c_i[i] + d_u[u],
#         a_p[p] + c_i[i] + d_u[u],
#         a_p[p] + tb,
#         c_i[i] + tb,
#         d_u[u] + tb,
#         a_p[p] + c_i[i] + tb,
#         a_p[p] + d_u[u] + tb,
#         c_i[i] + d_u[u] + tb,
#         a_p[p] + c_i[i] + d_u[u],
#         a_p[p] + c_i[i] + d_u[u] + tb
#     ]
#
#
# def csv2data1():
#     with open(OUT_DIR + '/' + '__a.pickle', 'rb') as pf:
#         a_p = pickle.load(pf)
#
#     with open(OUT_DIR + '/' + '__b.pickle', 'rb') as pf:
#         b_ut = pickle.load(pf)
#
#     with open(OUT_DIR + '/' + '__c.pickle', 'rb') as pf:
#         c_i = pickle.load(pf)
#
#     with open(OUT_DIR + '/' + '__d.pickle', 'rb') as pf:
#         d_u = pickle.load(pf)
#
#     train_data = [[]] * len(train)
#     test_data = [[]] * len(test)
#
#     for pos, v in enumerate(train):
#         train_data[pos] = v2data1(v, a_p, b_ut, c_i, d_u)
#
#     for pos, v in enumerate(test):
#         test_data[pos] = v2data1(v, a_p, b_ut, c_i, d_u)
#
#     train_data = np.array(train_data, dtype=float)
#     test_data = np.array(test_data, dtype=float)
#
#     return train_data, test_data


def csv2data():
    train_data, test_data = csv2data_()

    # train_tmp, test_tmp = csv2data1()
    # train_data = np.concatenate((train_data, train_tmp), axis=1)
    # test_data = np.concatenate((test_data, test_tmp), axis=1)

    return train_data, test_data


def save_result(test_probas):
    with open(OUT_DIR + '/' + ALG_NAME + '.csv', 'w') as f:
        f.write('sample_id,target\n')
        for pos, v in enumerate(test):
            r = test_probas[pos][1]
            f.write('%s,%s\n' % (v[f_sample_id], r))


def main():
    test_has_target = f_target < len(test[0])

    train_data, test_data = csv2data()

    train_labels = list(map(lambda v: 1 if v[f_target] else -1, train))
    train_data = Pool(data=train_data, label=train_labels)

    if test_has_target:
        test_labels = list(map(lambda v: 1 if v[f_target] else -1, test))
        test_data = Pool(data=test_data, label=test_labels)
    else:
        test_data = Pool(data=test_data)

    model = CatBoostClassifier(
        iterations=ITERATION_COUNT,
        learning_rate=ETA,
        depth=TREE_DEPTH,
        border_count=BORDER_COUNT,
        random_strength=RANDOM_STRENGTH)

    if test_has_target:
        model.fit(train_data, eval_set=test_data)
    else:
        model.fit(train_data)

    train_probas = model.predict_proba(train_data)
    test_probas = model.predict_proba(test_data)

    train_r = list(map(lambda p: sigmoid1(p[1]), train_probas))
    test_r = list(map(lambda p: sigmoid1(p[1]), test_probas))

    with open(OUT_DIR + '/' + '_10_train.pickle', 'wb') as f:
        pickle.dump(train_r, f)

    with open(OUT_DIR + '/' + '_10_test.pickle', 'wb') as f:
        pickle.dump(test_r, f)

    save_result(test_probas)


main()
