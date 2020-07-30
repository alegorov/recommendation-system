import os
import pickle
from utils import *
from catboost import CatBoostRegressor, Pool

SRC_DIR = 'data0'
ITERATION_COUNT = 588

# TREE_DEPTH = 8
# BORDER_COUNT = 256
# RANDOM_STRENGTH = 2.

ETA = 0.1

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

test_ids = list(map(lambda v: v[f_sample_id], test))


def get_r0(d, tp):
    ans = d

    for t_, p_ in tp:
        ans += p_ * e_t[t_]

    return ans


with open(OUT_DIR + '/' + '_7_d.pickle', 'rb') as pf:
    d_u = pickle.load(pf)

with open(OUT_DIR + '/' + '_7_e.pickle', 'rb') as pf:
    e_t = pickle.load(pf)


def calculate_r0(itr):
    for v in itr:
        u = v[f_user]
        d = d_u[u]

        tp = get_topic_probas(v)

        v[f_sample_id] = sigmoid(get_r0(d, tp))


calculate_r0(train)
calculate_r0(test)


def v2data_(v):
    return [
        # v[f_item],
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


class LoglossMetric(object):
    def get_final_error(self, error, weight):
        return error / weight

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers
        # (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.
        # weight parameter can be None.
        # Returns pair (error, weights sum)

        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        error_sum = 0.
        weight_sum = 0.

        for i, approx in enumerate(approxes[0]):
            w = 1. if weight is None else weight[i]
            weight_sum += w
            if target[i] > 999.:
                v = test[round(target[i] / 1000) - 1]
                y = 1. if v[f_target] else -1.
                error_sum -= w * log_sigmoid(y * (sigmoid1(approx + v[f_sample_id])))
            else:
                error_sum += w * (target[i] - approx) ** 2

        return error_sum, weight_sum


def save_result(test_r):
    with open(OUT_DIR + '/' + ALG_NAME + '.csv', 'w') as f:
        f.write('sample_id,target\n')
        for pos, sample_id in enumerate(test_ids):
            r = sigmoid(mclip(test_r[pos]))
            f.write('%s,%s\n' % (sample_id, r))


def main():
    test_has_target = f_target < len(test[0])

    train_data, test_data = csv2data()

    cat_features = list(range(7))

    train_labels = list(map(lambda v: v[f_target] - v[f_sample_id], train))
    train_data = Pool(data=train_data, label=train_labels, cat_features=cat_features)

    if test_has_target:
        test_labels = list(map(lambda x: 1000. * (x + 1), range(len(test))))
        test_data = Pool(data=test_data, label=test_labels, cat_features=cat_features)
    else:
        test_data = Pool(data=test_data, cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=ITERATION_COUNT,
        # depth=TREE_DEPTH,
        # border_count=BORDER_COUNT,
        # random_strength=RANDOM_STRENGTH,
        loss_function='RMSE',
        eval_metric=LoglossMetric(),
        learning_rate=ETA,
    )

    if test_has_target:
        model.fit(train_data, eval_set=test_data)
    else:
        model.fit(train_data)

    train_predict = model.predict(train_data)
    test_predict = model.predict(test_data)

    train_r = list(map(lambda pos: sigmoid1(train_predict[pos] + train[pos][f_sample_id]), range(len(train))))
    test_r = list(map(lambda pos: sigmoid1(test_predict[pos] + test[pos][f_sample_id]), range(len(test))))

    with open(OUT_DIR + '/' + '_7c-i_train.pickle', 'wb') as f:
        pickle.dump(train_r, f)

    with open(OUT_DIR + '/' + '_7c-i_test.pickle', 'wb') as f:
        pickle.dump(test_r, f)

    save_result(test_r)


main()
