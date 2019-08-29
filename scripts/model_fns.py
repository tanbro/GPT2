from functools import partial
from typing import Any, Union, Optional, Dict

import numpy as np
import tensorflow as tf

import model
import sample

from optimizers import create_train_op
from metric_fns import perplexity_metric


def model_fn(
    features: Union[tf.Tensor, Dict[str, Any]],
    labels: Union[tf.Tensor, Dict[str, Any]],
    mode: tf.estimator.ModeKeys=None,
    params:Optional[Dict[str, Any]]=None,
    config:Union[tf.contrib.training.HParams, Dict[str, Any]]=None,
) -> tf.estimator.EstimatorSpec:
    """包装 gpt-2 模型定义的模型函数

    Parameters
    ----------

    features :
        This is the first item returned from the input_fn passed to train, evaluate, and predict.
        This should be a single tf.Tensor or dict of same.

    labels :
        This is the second item returned from the input_fn passed to train, evaluate, and predict.
        This should be a single tf.Tensor or dict of same (for multi-head models).
        If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed. If the model_fn's signature does not accept mode, the model_fn must still be able to handle labels=None.

    mode :
        Optional. Specifies if this training, evaluation or prediction.
        See :cls:`tf.estimator.ModeKeys`.

    params :
        Optional dict of hyperparameters.
        Will receive what is passed to Estimator in params parameter.
        This allows to configure Estimators from hyper parameter tuning.

        .. note:: 此处，我们使用两级别 Dict 装载配置项！

    config:
        Optional estimator.RunConfig object.
        Will receive what is passed to Estimator as its config parameter, or a default value. Allows setting up things in your model_fn based on configuration such as num_ps_replicas, or model_dir.
    
    Returns
    -------
        tf.estimator.EstimatorSpec
    """
    if params is None:
        params = {}
    if config is None:
        config = {}

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        output = model.model(
            hparams=tf.contrib.training.HParams(**params['hparams']),
            X=features,
            reuse=tf.compat.v1.AUTO_REUSE,
        )
        loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output["logits"],
            labels=labels
        )
        loss = tf.reduce_mean(loss_batch)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = create_train_op(loss, params['optimizer'])
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss,
            eval_metric_ops=perplexity_metric(loss_batch)
        )

    if mode == tf.estimator.ModeKeys.PREDICT:

        if not "top_k" in params.keys():
            params["top_k"] = 0

        output = sample.sample_sequence(
            params=params, length=params["n_ctx"],
            context=features,
            batch_size=params["batch_size"],
            temperature=1.0, top_k=params["top_k"]
        )

        predictions = {
            "tokens": output
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
