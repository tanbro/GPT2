"""
train from scratch.
"""

import argparse
import json
import logging
import os
import signal
import sys
from time import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import yaml
from dotenv import load_dotenv
from wasabi import table

import tensorflow as tf
from model import default_hparams
from input_fns import input_fn
from model_fns import model_fn


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-l', '--log-level', type=str,
        choices=[k for k in logging._nameToLevel.keys()], default='INFO',
        help='日志级别 (default=%(default)s)'
    )
    parser.add_argument(
        '-d', '--disable-eager-execution', action='store_true',
        help='禁止 Eager Execution (default=%(default)s)'
    )
    parser.add_argument(
        '-e', '--env-path', type=Path,
        help='.env 文件路径，用于定义环境变量 (default=%(default)s)'
    )
    parser.add_argument(
        '--disable-dotenv', action='store_true',
        help='禁止从 .env 文件加载环境变量 (default=%(default)s)'
    )
    parser.add_argument(
        'config_file',
        type=Path, nargs='?', default='config.yml',
        help='配置文件 (default=%(default)s)'
    )
    arguments = parser.parse_args()
    return arguments


def main(args: argparse.Namespace):
    log_level = logging._nameToLevel[args.log_level]
    logger = logging.getLogger(__name__)
    logger.setLevel(logging._nameToLevel[args.log_level])

    if not args.disable_dotenv:
        logger.info('Load environment varaiblas: %s', args.env_path)
        load_dotenv(dotenv_path=args.env_path)

    logger.info("TensorFlow version: %s", tf.version.VERSION)

    if not args.disable_eager_execution:
        tf.compat.v1.enable_eager_execution()
    logger.info("Eager execution: %s", tf.executing_eagerly())

    logger.info('Configure file: %s', args.config_file)
    conf = yaml.full_load(args.config_file.open())
    logger.info('Configurations: %s', conf)

    run_config = tf.estimator.RunConfig(**conf['estimator']['run'])
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=conf['estimator'],
    )

    epochs = conf['epochs']
    start_ts = time()

    for epoch in range(epochs):
        train_ts = time()
        estimator.train(
            input_fn=partial(input_fn, conf['estimator']['train_input'], conf['estimator']['hparams']),
            steps=conf['estimator'].get('train_steps'),
        )

        eval_ts = time()
        eval_result = estimator.evaluate(
            input_fn=partial(input_fn, conf['estimator']['eval_input'], conf['estimator']['hparams'], eval=True),
            steps=conf['estimator'].get('eval_steps'),
        )

        ts = time()
        dur_info = {
            'total': timedelta(seconds=ts-start_ts),
            'epoch_train': timedelta(seconds=eval_ts-train_ts),
            'epoch_eval': timedelta(seconds=ts-eval_ts),
        }
        duration_tab = table(
            [list(dur_info.values())],
            header=list(dur_info.keys()),
            divider=True
        )

        eval_result_tab = table(
            [list(eval_result.values())],
            header=list(eval_result.keys()),
            divider=True
        )

        logger.info(
            '''
============================================

Epoch [%d/%d] completed

duration:

%s

eval results:

%s

============================================

''',
            epoch+1, epochs,
            duration_tab, eval_result_tab
        )


if __name__ == '__main__':
    args = parse_args()
    try:
        sys.exit(main(args))
    except KeyboardInterrupt as exception:
        print(exception, file=sys.stderr)
        sys.exit(signal.SIGINT)
