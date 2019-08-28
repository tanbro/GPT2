import os
import warnings
from glob import glob, iglob
from functools import partial

from tqdm import tqdm

import tensorflow as tf


def _sample_fn(min_rate: tf.Tensor, max_rate: tf.Tensor, x: tf.Tensor):
    sz = tf.size(x)
    min_sz = tf.cast(
        tf.math.ceil(
            tf.cast(sz, tf.float32) * min_rate
        ),
        tf.int32
    )
    max_sz = tf.cast(
        tf.math.round(
            tf.cast(sz, tf.float32) * max_rate
        ),
        tf.int32
    )
    min_sz = tf.cond(
        tf.math.equal(min_sz, tf.convert_to_tensor(0, tf.int32)),
        true_fn=lambda: tf.add(min_sz, tf.convert_to_tensor(1, tf.int32)),
        false_fn=lambda: min_sz,
    )
    max_sz = tf.cond(
        tf.math.less_equal(max_sz, min_sz),
        true_fn=lambda: tf.add(min_sz, tf.convert_to_tensor(1, tf.int32)),
        false_fn=lambda: max_sz,
    )
    amount = tf.random.uniform(
        [], minval=min_sz, maxval=max_sz, dtype=tf.dtypes.int32
    )
    pos = tf.random.uniform([], maxval=sz-amount, dtype=tf.dtypes.int32)
    r1 = tf.range(pos, pos + amount)
    r2 = tf.range(pos + 1, pos + amount + 1)
    r1 = tf.reshape(r1, [amount])  # Somehow, this makes the compiler happy
    # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
    r2 = tf.reshape(r2, [amount])
    v1 = tf.gather(x, r1)
    v2 = tf.gather(x, r2)
    return v1, v2


def _parse_fn(example_proto):
    features = {'text': tf.VarLenFeature(tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return tf.sparse.to_dense(parsed_features['text'])


def input_fn(
    params: dict,
    eval: bool = False,
):
    data_files = glob(params['data_files'])
    dataset = (
        tf.data.Dataset.from_tensor_slices(data_files)
        .interleave(
            lambda x: tf.data.TFRecordDataset(x).map(_parse_fn),
            cycle_length = params['interleave_cycle_length']
        )
        .filter(lambda x: tf.size(x) > tf.convert_to_tensor(1, tf.int32))
        .map(
            partial(
                _sample_fn,
                tf.convert_to_tensor(params['tokens_sample_rate_min'], tf.float32),
                tf.convert_to_tensor(params['tokens_sample_rate_max'], tf.float32)
            )
        )
        .shuffle(buffer_size=params['shuffle_buffer_size'])
        .batch(
            batch_size=1, # 由于变长 Tensor， 批只能是 1 !!!
            # batch_size=params['batch_size']
        )
        # .prefetch(buffer_size=params['prefetch_buffer_size'])
    )
    return dataset
