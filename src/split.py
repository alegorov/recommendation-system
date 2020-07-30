import random
from utils import *

TEST_PART = 0.1

random.seed(783689140)


def get_item_set(data):
    return get_element_set(data, [f_item])


def get_publisher_set(data):
    return get_element_set(data, [f_publisher])


def get_user_set(data):
    return get_element_set(data, [f_user])


def get_topic_set(data):
    return get_element_set(data, [f_topic_0, f_topic_1, f_topic_2, f_topic_3, f_topic_4])


def set2map(s):
    a = list(s)
    a.sort()
    m = {}
    for i, ai in enumerate(a):
        m[ai] = i
    return m


def correct_field(field, field_map, v):
    if v[field] in field_map:
        v[field] = field_map[v[field]]
        return True
    return False


def save_dst(maps, fpath, dst):
    item_map, publisher_map, user_map, topic_map = maps

    with open(fpath, 'w') as f:
        f.write(
            'sample_id,item,publisher,user,topic_0,topic_1,topic_2,topic_3,topic_4,weight_0,weight_1,weight_2,weight_3,weight_4,target\n')
        for v0 in dst:
            v = v0.copy()

            if not correct_field(f_item, item_map, v):
                continue
            if not correct_field(f_publisher, publisher_map, v):
                continue
            if not correct_field(f_user, user_map, v):
                continue
            if not correct_field(f_topic_0, topic_map, v):
                continue
            if not correct_field(f_topic_1, topic_map, v):
                continue
            if not correct_field(f_topic_2, topic_map, v):
                continue
            if not correct_field(f_topic_3, topic_map, v):
                continue
            if not correct_field(f_topic_4, topic_map, v):
                continue

            f.write(','.join(map(str, v)) + '\n')


def main(src_dir, dst_dir):
    src = open_csv(src_dir + '/train.csv')

    len1 = round((1.0 - TEST_PART) * len(src))

    random.shuffle(src)

    dst1 = src[:len1]
    dst2 = src[len1:]

    item_map = set2map(get_item_set(dst1))
    publisher_map = set2map(get_publisher_set(dst1))
    user_map = set2map(get_user_set(dst1))
    topic_map = set2map(get_topic_set(dst1))

    maps = [item_map, publisher_map, user_map, topic_map]

    save_dst(maps, dst_dir + '/train.csv', dst1)
    save_dst(maps, dst_dir + '/test.csv', dst2)

    train = open_csv(dst_dir + '/train.csv')
    test = open_csv(dst_dir + '/test.csv')

    print('test part: ', len(test) / (len(train) + len(test)))


main('data0', 'data1')
# main('data1', 'data2')
