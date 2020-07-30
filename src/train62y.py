import os
import pickle
from utils import *
from catboost import CatBoostClassifier, Pool

SRC_DIR = 'data0'
ITERATION_COUNT = 4265

FEATURE_COUNT = 100

# TREE_DEPTH = 1
# BORDER_COUNT = 256
# RANDOM_STRENGTH = 1060.

ETA = 0.1

ALG_NAME = str(os.path.splitext(os.path.basename(__file__))[0].split('-')[0])
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

with open(OUT_DIR + '/__' + ALG_NAME[:-1] + 'x__aa.pickle', 'rb') as pf:
    aa_tf = np.transpose(pickle.load(pf)[:FEATURE_COUNT, :], (1, 0))

with open(OUT_DIR + '/__' + ALG_NAME[:-1] + 'x__bb.pickle', 'rb') as pf:
    bb_uf = np.transpose(pickle.load(pf)[:FEATURE_COUNT, :], (1, 0))


def v2data_(v):
    u = v[f_user]

    tp = get_topic_probas(v)

    aa = np.zeros(FEATURE_COUNT, dtype=float)

    for t_, p_ in tp:
        aa += p_ * aa_tf[t_]

    bb = bb_uf[u]

    return [aa[-1], bb[-1]] + (aa * bb).tolist()


def csv2data():
    train_data = [[]] * len(train)
    test_data = [[]] * len(test)

    for pos, v in enumerate(train):
        train_data[pos] = v2data_(v)

    for pos, v in enumerate(test):
        test_data[pos] = v2data_(v)

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
        # depth=TREE_DEPTH,
        # border_count=BORDER_COUNT,
        # random_strength=RANDOM_STRENGTH,
        iterations=ITERATION_COUNT,
        learning_rate=ETA
    )

    if test_has_target:
        model.fit(train_data, eval_set=test_data)
    else:
        model.fit(train_data)

    test_probas = model.predict_proba(test_data)

    save_result(test_probas)


main()
