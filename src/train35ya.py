import os
import pickle
from utils import *
from catboost import CatBoostRegressor, Pool

SRC_DIR = 'data0'
ITERATION_COUNT = 388

FEATURE_COUNT = 100

# TREE_DEPTH = 1
# RANDOM_STRENGTH = 8.
# BORDER_COUNT = 256

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

test_ids = list(map(lambda v: v[f_sample_id], test))


def get_r0(c):
    return c


with open(OUT_DIR + '/' + '_4i_c.pickle', 'rb') as pf:
    c_i = pickle.load(pf)


def calculate_r0(itr):
    for v in itr:
        i = v[f_item]
        c = c_i[i]

        v[f_sample_id] = get_r0(c)


calculate_r0(train)
calculate_r0(test)


def set_sigmoid_r0(dst):
    for v in dst:
        v[f_sample_id] = sigmoid(v[f_sample_id])


set_sigmoid_r0(train)
set_sigmoid_r0(test)

with open(OUT_DIR + '/__' + ALG_NAME[:-2] + 'x__aa.pickle', 'rb') as pf:
    aa_tf = np.transpose(pickle.load(pf)[:FEATURE_COUNT, :], (1, 0))

with open(OUT_DIR + '/__' + ALG_NAME[:-2] + 'x__bb.pickle', 'rb') as pf:
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


def save_result(test_predict):
    with open(OUT_DIR + '/' + ALG_NAME + '.csv', 'w') as f:
        f.write('sample_id,target\n')
        for pos, sample_id in enumerate(test_ids):
            r = test_predict[pos] + test[pos][f_sample_id]
            r = min(max(r, 0.), 1.)
            f.write('%s,%s\n' % (sample_id, r))


def main():
    test_has_target = f_target < len(test[0])

    train_data, test_data = csv2data()

    train_labels = list(map(lambda v: v[f_target] - v[f_sample_id], train))
    train_data = Pool(data=train_data, label=train_labels)

    if test_has_target:
        test_labels = list(map(lambda x: 1000. * (x + 1), range(len(test))))
        test_data = Pool(data=test_data, label=test_labels)
    else:
        test_data = Pool(data=test_data)

    model = CatBoostRegressor(
        iterations=ITERATION_COUNT,
        # depth=TREE_DEPTH,
        # random_strength=RANDOM_STRENGTH,
        # border_count=BORDER_COUNT,
        loss_function='RMSE',
        eval_metric=LoglossMetric(),
        learning_rate=ETA,
    )

    if test_has_target:
        model.fit(train_data, eval_set=test_data)
    else:
        model.fit(train_data)

    save_result(model.predict(test_data))


main()
