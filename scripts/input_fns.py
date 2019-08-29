from functools import partial

import tensorflow as tf


def _fix_length(t: tf.Tensor, length: tf.Tensor)->tf.Tensor:
    """不够的在末尾补零，超出的截断末尾
    
    Parameters
    ----------

    t : tf.Tensor
        IDs of text

    length : tf.Tensor
        expected fixed length, an interger tensor
    
    Returns
    -------
    tf.Tensor
        fixed tensor of IDs
    """
    sz = tf.size(t)
    return tf.cond(
        length > sz,
        true_fn=lambda: tf.pad(t, tf.stack([tf.stack([0, length-sz])])),
        false_fn=lambda: tf.slice(t, [0], [length])
    )


def _sample_fn(n_ctx, min_rate: tf.Tensor, max_rate: tf.Tensor, x: tf.Tensor):
    """样本处理函数

    在一个文本样本中，采样若干 tokens 作为输入数据，输入数据的内容向后偏移一个 token 作为标注数据
    """
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
    # Somehow, this makes the compiler happy
    # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
    r1 = tf.reshape(r1, [amount])
    r2 = tf.reshape(r2, [amount])
    v1 = tf.gather(x, r1)
    v2 = tf.gather(x, r2)
    return _fix_length(v1, n_ctx), _fix_length(v2, n_ctx)


def _parse_fn(example_proto):
    features = {'text': tf.VarLenFeature(tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return tf.sparse.to_dense(parsed_features['text'])


def input_fn(
    params: dict,
    hparams: dict,
    eval: bool = False,
):
    data_files = tf.matching_files(params['data_files'])
    dataset = (
        tf.data.Dataset.from_tensor_slices(data_files)
        .interleave(
            lambda x: tf.data.TFRecordDataset(x).map(_parse_fn),
            cycle_length=1
        )
        .filter(lambda x: tf.size(x) > 1)
        .repeat(count=params.get('repeat_count', 1))
        .shuffle(buffer_size=params['shuffle_buffer_size'])
        .map(partial(
            _sample_fn,
            tf.constant(hparams['n_ctx']),
            tf.constant(params['tokens_sample_rate_min']),
            tf.constant(params['tokens_sample_rate_max'])
        ))
        .batch(batch_size=params['batch_size'])
        .prefetch(buffer_size=params['prefetch_buffer_size'])
    )
    return dataset
