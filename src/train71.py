import os
import pickle
from utils import *
from catboost import CatBoostRegressor, Pool

SRC_DIR = 'data0'
ITERATION_COUNT = 636

FEATURE_COUNT = 100

# TREE_DEPTH = 1
# RANDOM_STRENGTH = 8.
# BORDER_COUNT = 256

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

test_ids = list(map(lambda v: v[f_sample_id], test))

global_r0 = sigmoid1(sum(map(lambda v: v[f_target], train)) / len(train))


def set_r0(dst):
    for v in dst:
        v[f_sample_id] = global_r0


set_r0(train)
set_r0(test)


def set_sigmoid_r0(dst):
    for v in dst:
        v[f_sample_id] = sigmoid(v[f_sample_id])


set_sigmoid_r0(train)
set_sigmoid_r0(test)


def load_aa_bb(names):
    aa = [[]] * len(names)
    bb = [[]] * len(names)

    for i, name in enumerate(names):
        with open(OUT_DIR + '/__' + name + '__aa.pickle', 'rb') as pf:
            aa[i] = pickle.load(pf)[FEATURE_COUNT - 1].tolist()

        with open(OUT_DIR + '/__' + name + '__bb.pickle', 'rb') as pf:
            bb[i] = pickle.load(pf)[FEATURE_COUNT - 1].tolist()

    aa = np.array(aa, dtype=float)
    bb = np.array(bb, dtype=float)

    aa = np.transpose(aa, (1, 0))
    bb = np.transpose(bb, (1, 0))

    return aa, bb


i_names = [
    'train18x',
    # 'train21x',
    'train24x',
    'train27x',
    'train2x',
    'train30x',
    'train33x',
    'train36x',
    'train39x',
    'train42x',
    'train45x',
    'train48x',
    'train51x',
    'train54x',
    # 'train57x',
    'train5x',
    'train60x',
    'train6x',
]

p_names = [
    'train12x',
    'train14x',
    'train16x',
    'train19x',
    'train22x',
    'train25x',
    # 'train28x',
    'train31x',
    'train34x',
    # 'train37x',
    'train40x',
    'train43x',
    'train46x',
    'train49x',
    'train52x',
    # 'train55x',
    # 'train58x',
    'train61x',
]

t_names = [
    'train13x',
    'train15x',
    'train17x',
    # 'train20x',
    # 'train23x',
    'train26x',
    'train29x',
    'train32x',
    # 'train35x',
    # 'train38x',
    'train41x',
    'train44x',
    'train47x',
    'train50x',
    'train53x',
    'train56x',
    'train59x',
    # 'train62x',
]

i_aa_if, i_bb_uf = load_aa_bb(i_names)
p_aa_pf, p_bb_uf = load_aa_bb(p_names)
t_aa_tf, t_bb_uf = load_aa_bb(t_names)


def v2data_(v):
    i = v[f_item]
    p = v[f_publisher]
    u = v[f_user]

    tp = get_topic_probas(v)

    i_aa = i_aa_if[i]
    i_bb = i_bb_uf[u]

    p_aa = p_aa_pf[p]
    p_bb = p_bb_uf[u]

    t_aa = np.zeros(len(t_names), dtype=float)
    for t_, p_ in tp:
        t_aa += p_ * t_aa_tf[t_]
    t_bb = t_bb_uf[u]

    aa = i_aa.tolist() + p_aa.tolist() + t_aa.tolist()
    bb = i_bb.tolist() + p_bb.tolist() + t_bb.tolist()

    aa_bb = (i_aa * i_bb).tolist() + (p_aa * p_bb).tolist() + (t_aa * t_bb).tolist()

    return aa + bb + aa_bb


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
