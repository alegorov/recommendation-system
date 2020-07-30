import os
import pickle
from utils import *
from catboost import CatBoostClassifier, Pool

SRC_DIR = 'data0'
ITERATION_COUNT = 507

# TREE_DEPTH = 8
# BORDER_COUNT = 256
# RANDOM_STRENGTH = 2.

ETA = 0.3

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


def v2data_(v):
    return [
        v[f_item],
        v[f_publisher],
        v[f_user],
        v[f_topic_0],
        v[f_topic_1],
        v[f_topic_2],
        v[f_topic_3],
        v[f_topic_4],
        v[f_weight_0],
        v[f_weight_1],
        v[f_weight_2],
        v[f_weight_3],
        v[f_weight_4],
    ]


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

    cat_features = list(range(8))

    train_labels = list(map(lambda v: 1 if v[f_target] else -1, train))
    train_data = Pool(data=train_data, label=train_labels, cat_features=cat_features)

    if test_has_target:
        test_labels = list(map(lambda v: 1 if v[f_target] else -1, test))
        test_data = Pool(data=test_data, label=test_labels, cat_features=cat_features)
    else:
        test_data = Pool(data=test_data, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=ITERATION_COUNT,
        # depth=TREE_DEPTH,
        # border_count=BORDER_COUNT,
        # random_strength=RANDOM_STRENGTH,
        learning_rate=ETA)

    if test_has_target:
        model.fit(train_data, eval_set=test_data)
    else:
        model.fit(train_data)

    train_probas = model.predict_proba(train_data)
    test_probas = model.predict_proba(test_data)

    train_r = list(map(lambda p: sigmoid1(p[1]), train_probas))
    test_r = list(map(lambda p: sigmoid1(p[1]), test_probas))

    with open(OUT_DIR + '/' + '_9_train.pickle', 'wb') as f:
        pickle.dump(train_r, f)

    with open(OUT_DIR + '/' + '_9_test.pickle', 'wb') as f:
        pickle.dump(test_r, f)

    # save_result(test_probas)


main()
