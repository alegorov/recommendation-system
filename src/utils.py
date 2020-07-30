import numpy as np
import math

f_sample_id = 0
f_item = 1
f_publisher = 2
f_user = 3
f_topic_0 = 4
f_topic_1 = 5
f_topic_2 = 6
f_topic_3 = 7
f_topic_4 = 8
f_weight_0 = 9
f_weight_1 = 10
f_weight_2 = 11
f_weight_3 = 12
f_weight_4 = 13
f_target = 14


def open_csv(fname):
    with open(fname) as f:
        a = [s for s in map(lambda line: line.strip(), f) if s]
    a.pop(0)
    for i, s in enumerate(a):
        v = list(map(int, s.split(',')))
        a[i] = v
    return a


def open_pred_csv(fname):
    with open(fname) as f:
        a = [s for s in map(lambda line: line.strip(), f) if s]
    a.pop(0)
    for i, s in enumerate(a):
        v = s.split(',')
        v[0] = int(v[0])
        v[1] = float(v[1])
        a[i] = v
    return a


def get_element_set(data, fields):
    a = set()
    for field in fields:
        a |= set(map(lambda x: x[field], data))
    return a


def get_element_count(data, fields):
    a = len(get_element_set(data, fields))
    for field in fields:
        for v in data:
            if not ((v[field] >= 0) and (v[field] < a)):
                raise Exception('not ((v[field] >= 0) and (v[field] < a))')
    return a


def get_item_count(data):
    return get_element_count(data, [f_item])


def get_publisher_count(data):
    return get_element_count(data, [f_publisher])


def get_user_count(data):
    return get_element_count(data, [f_user])


def get_topic_count(data):
    return get_element_count(data, [f_topic_0, f_topic_1, f_topic_2, f_topic_3, f_topic_4])


def get_topic_vect(v, topic_count):
    t = [0.] * topic_count

    t[v[f_topic_0]] = v[f_weight_0] / 100
    t[v[f_topic_1]] = v[f_weight_1] / 100
    t[v[f_topic_2]] = v[f_weight_2] / 100
    t[v[f_topic_3]] = v[f_weight_3] / 100
    t[v[f_topic_4]] = v[f_weight_4] / 100

    return np.array(t, float)


def get_topic_probas(v):
    s = v[f_weight_0] + v[f_weight_1] + v[f_weight_2] + v[f_weight_3] + v[f_weight_4]
    if s < 1:
        s = 1

    return (
        (v[f_topic_0], v[f_weight_0] / s),
        (v[f_topic_1], v[f_weight_1] / s),
        (v[f_topic_2], v[f_weight_2] / s),
        (v[f_topic_3], v[f_weight_3] / s),
        (v[f_topic_4], v[f_weight_4] / s),
    )


def logloss(y_true, y_pred):
    eps = 1e-4

    y_pred = max(y_pred, eps)
    y_pred = min(y_pred, 1. - eps)

    return -y_true * math.log(y_pred) - (1 - y_true) * math.log(1. - y_pred)


def score(loss):
    if loss > 0.5:
        return 0.
    return min(150., (200. * (0.5 - loss)) ** 2)


def sigmoid(x):
    if x >= 0.:
        return 1. / (1. + math.exp(-x))
    e = math.exp(x)
    return e / (1. + e)


def log_sigmoid(x):
    if x >= 0.:
        return -math.log(1. + math.exp(-x))
    return x - math.log(1. + math.exp(x))


def sigmoid1(y):
    eps = 1e-4

    y = max(y, eps)
    y = min(y, 1. - eps)

    if y <= 0.5:
        return -math.log((1. - y) / y)
    return math.log(y / (1. - y))


def mclip(m):
    # bound = -sigmoid1(1e-4)
    bound = 9.21024036697585
    return max(-bound, min(bound, m))


def np_sigmoid(x):
    return 1. / (1. + np.exp(np.clip(-x, -300., 300.)))
