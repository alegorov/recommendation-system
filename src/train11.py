import os
import pickle
from utils import *
import random
import tensorflow as tf

SRC_DIR = 'data1'
ITERATION_COUNT = 117

NUM_HIDDEN = 100
LOGITS_AMPLITUDE = 20.
REGULARIZER_SCALE = 0.2
DROPOUT_RATE = 0.3

BATCH_SIZE = 100

# AdamOptimizer:
# ETA = 0.0000005

# GradientDescentOptimizer:
ETA = 0.0001

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


def v2data_(v, a_p, b_ut, b0_ut, b1_ut, c_i, d_u, e_t):
    i = v[f_item]
    u = v[f_user]
    p = v[f_publisher]

    t = get_topic_vect(v, topic_count)
    tp = get_topic_probas(v)

    e = 0.
    for t_, p_ in tp:
        e += p_ * e_t[t_]

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
        a_p[p] + c_i[i] + d_u[u],
        e,
        e + a_p[p],
        e + c_i[i],
        e + d_u[u],
        e + a_p[p] + c_i[i],
        e + a_p[p] + d_u[u],
        e + c_i[i] + d_u[u],
        e + a_p[p] + c_i[i] + d_u[u],
    ]


def csv2data_():
    n_p = np.zeros(publisher_count, dtype=float)
    n_i = np.zeros(item_count, dtype=float)
    n_u = np.zeros(user_count, dtype=float)
    n_t = np.zeros(topic_count, dtype=float)

    a_p = np.zeros(publisher_count, dtype=float)
    b_ut = np.zeros((user_count, topic_count), dtype=float)
    b0_ut = np.zeros((user_count, topic_count), dtype=float)
    b1_ut = np.zeros((user_count, topic_count), dtype=float)
    c_i = np.zeros(item_count, dtype=float)
    d_u = np.zeros(user_count, dtype=float)
    e_t = np.zeros(topic_count, dtype=float)

    for v in train:
        i = v[f_item]
        u = v[f_user]
        p = v[f_publisher]

        n_p[p] += 1.
        n_i[i] += 1.
        n_u[u] += 1.

        y = 1. if v[f_target] else -1.

        t = get_topic_vect(v, topic_count)
        tp = get_topic_probas(v)

        a_p[p] += y

        if v[f_target]:
            b_ut[u] += t
            b1_ut[u] += t
        else:
            b_ut[u] -= t
            b0_ut[u] += t

        c_i[i] += y
        d_u[u] += y

        dn = 1 / len(tp)

        for t_, p_ in tp:
            n_t[t_] += dn
            e_t[t_] += p_ * y

    a_p /= n_p
    b_ut /= n_u[:, np.newaxis]
    b0_ut /= n_u[:, np.newaxis]
    b1_ut /= n_u[:, np.newaxis]
    c_i /= n_i
    d_u /= n_u
    e_t /= n_t

    train_data = [[]] * len(train)
    test_data = [[]] * len(test)

    for pos, v in enumerate(train):
        train_data[pos] = v2data_(v, a_p, b_ut, b0_ut, b1_ut, c_i, d_u, e_t)

    for pos, v in enumerate(test):
        test_data[pos] = v2data_(v, a_p, b_ut, b0_ut, b1_ut, c_i, d_u, e_t)

    train_data = np.array(train_data, dtype=float)
    test_data = np.array(test_data, dtype=float)

    return train_data, test_data


def csv2data():
    train_data, test_data = csv2data_()

    x = np.mean(train_data, axis=0)

    train_data -= x
    test_data -= x

    x = np.sqrt(np.mean(train_data * train_data, axis=0))

    train_data /= x
    test_data /= x

    return list(train_data.tolist()), list(test_data.tolist())


def save_result(test_r):
    with open(OUT_DIR + '/' + ALG_NAME + '.csv', 'w') as f:
        f.write('sample_id,target\n')
        for pos, v in enumerate(test):
            r = sigmoid(mclip(test_r[pos]))
            f.write('%s,%s\n' % (v[f_sample_id], r))


class Iter:
    def __init__(self, x, y, do_shuffle):
        self.x = x
        self.y = y
        self.do_shuffle = do_shuffle
        self.indices = list(range(len(x)))
        self.pos = 0
        self.reset()

    def reset(self):
        self.pos = 0
        if self.do_shuffle:
            random.shuffle(self.indices)

    def next1(self):
        if self.pos >= len(self.indices):
            return None
        i = self.indices[self.pos]
        if self.y:
            v = self.y[i]
            result = [self.x[i], 1. if v[f_target] else -1.]
        else:
            result = [self.x[i], None]
        self.pos += 1
        return result

    def next(self):
        x = []
        y = []
        for _ in range(BATCH_SIZE):
            itm = self.next1()
            if not itm:
                break
            x.append(itm[0])
            y.append(itm[1])
        if not x:
            return [None, None]
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return [x, y]


def create_model(data_size):
    is_training = tf.placeholder(tf.bool, [])
    inputs = tf.placeholder(tf.float32, [None, data_size])

    net = inputs
    net = tf.layers.dense(net, NUM_HIDDEN)
    net = tf.nn.tanh(net)

    if DROPOUT_RATE > 0.:
        net = tf.layers.dropout(net, DROPOUT_RATE, training=is_training)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE)
    net = tf.layers.dense(net, 1, kernel_regularizer=regularizer)
    net = tf.reshape(net, [-1])
    logits = LOGITS_AMPLITUDE * tf.nn.tanh((1. / LOGITS_AMPLITUDE) * net)

    return is_training, inputs, logits


def get_logits(itr, session, is_training, inputs, labels, logits):
    itr.reset()

    ans = [0.] * len(itr.x)
    pos = 0

    while True:
        inputs0, labels0 = itr.next()

        if inputs0 is None:
            break

        feed = {is_training: False,
                inputs: inputs0,
                labels: labels0}

        logits0 = session.run(logits, feed)

        for r in logits0:
            ans[pos] = r
            pos += 1

    if pos != len(itr.x):
        raise Exception('pos != len(itr.x)')

    return ans


def do_train():
    train_data, test_data = csv2data()

    train_iter = Iter(train_data, train, do_shuffle=True)

    if f_target < len(test[0]):
        test_iter = Iter(test_data, test, do_shuffle=False)
    else:
        test_iter = Iter(test_data, None, do_shuffle=False)

    graph = tf.Graph()
    with graph.as_default():
        is_training, inputs, logits = create_model(len(train_iter.x[0]))
        labels = tf.placeholder(tf.float32, [None])

        m = logits * labels
        cost_test = tf.reduce_mean(tf.log(1. + tf.exp(-m)))
        cost = cost_test

        cost += tf.losses.get_regularization_loss()

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=ETA).minimize(cost)
        # optimizer = tf.train.AdamOptimizer(learning_rate=ETA,
        #                                    beta1=0.,
        #                                    beta2=0.1,
        #                                    epsilon=1e-30).minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        # Run the initializer
        session.run(init)

        best_cost = 1e10
        best_epoch = -1

        for curr_epoch in range(1, ITERATION_COUNT + 1):
            train_cost = 0.
            num_examples = 0

            train_iter.reset()

            while True:
                inputs0, labels0 = train_iter.next()

                if inputs0 is None:
                    break

                batch_size0 = inputs0.shape[0]

                feed = {is_training: True,
                        inputs: inputs0,
                        labels: labels0}

                cost0, _ = session.run([cost, optimizer], feed)

                num_examples += batch_size0
                train_cost += cost0 * batch_size0

            train_cost /= num_examples

            if test_iter.y:
                test_iter.reset()

                test_cost = 0.
                num_examples = 0

                while True:
                    inputs0, labels0 = test_iter.next()

                    if inputs0 is None:
                        break

                    batch_size0 = inputs0.shape[0]

                    feed = {is_training: False,
                            inputs: inputs0,
                            labels: labels0}

                    cost0 = session.run(cost_test, feed)

                    num_examples += batch_size0
                    test_cost += cost0 * batch_size0

                test_cost /= num_examples

                if test_cost < best_cost:
                    best_cost = test_cost
                    best_epoch = curr_epoch

                print_str = '%4d: train = %.6f, test = %.6f, best = %.6f (%s)' % (
                    curr_epoch, train_cost, test_cost, best_cost, best_epoch)
            else:
                print_str = '%4d: train = %.6f' % (curr_epoch, train_cost)

            print(print_str, flush=True)

        train_iter = Iter(train_data, train, do_shuffle=False)

        train_r = get_logits(train_iter, session, is_training, inputs, labels, logits)
        test_r = get_logits(test_iter, session, is_training, inputs, labels, logits)

        return train_r, test_r


def main():
    train_r, test_r = do_train()

    with open(OUT_DIR + '/' + '_11_train.pickle', 'wb') as f:
        pickle.dump(train_r, f)

    with open(OUT_DIR + '/' + '_11_test.pickle', 'wb') as f:
        pickle.dump(test_r, f)

    save_result(test_r)


main()
