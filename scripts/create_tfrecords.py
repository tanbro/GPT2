"""利用 BPE 模型将一批 JSON Lines 语料转为 *.tfrecords 文件

输入语料是 JSON Line 格式，其每一行都是一本 JSON 对象，该对象包含名为 text 的文本属性，其内容是语料平面文本。

输出语料是 HDF5 格式的数组，其数组元素是 Tensorflow Feature 对象，在其 text 属性中存放对应文本的 Tokens ID 数组。
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import timedelta
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from time import time

import numpy as np
import sentencepiece as spm
from tqdm.auto import tqdm

import tensorflow as tf


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


_g_args: argparse.Namespace = None
_g_sp = spm.SentencePieceProcessor()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--bpe-model', type=Path,
                        required=True, help='BPE 模型文件的路径')
    parser.add_argument('-x', '--context-max', type=int, default=1024,
                        help='语料最大 Tokens 长度。语料超出部分将被截断 (default=%(default)s)')
    parser.add_argument('-n', '--context-min', type=int, default=16,
                        help='语料最小 Tokens 长度。不足最小长度的语料将被忽略 (default=%(default)s)')
    parser.add_argument('-p', '--pool-processes', type=int,
                        help='用于并发处理语料的进程数。默认为 CPU 核心数')
    parser.add_argument('-c', '--pool-chunk', type=int, default=1,
                        help='用于并发处理语料的每进程任务分块数 (default=%(default)s)')
    parser.add_argument('-l', '--log-level', type=str,
                        choices=[k for k in logging._nameToLevel.keys()], default='INFO',
                        help='日志级别 (default=%(default)s)')
    parser.add_argument('--hide-progress', action='store_true',
                        help='隐藏进度条 (default=%(default)s)')
    parser.add_argument('input', type=Path, nargs=1,
                        help='待处理的 JSON Lines 格式输入语料文件路径')
    parser.add_argument('output', type=Path, nargs='?',
                        help='输出文件路径。默认输出到与输出文件基础名一致、后缀名为 ".tfrecords" 的 H5DF 文件')
    arguments = parser.parse_args()
    return arguments


def pool_func(line):
    line = line.strip()
    if not line:
        return None
    d = json.loads(line)
    text = d['text']
    ids = np.array(_g_sp.encode_as_ids(text), np.int32)
    length = ids.shape[0]
    if length < _g_args.context_min:
        return None
    ids = ids[:_g_args.context_max]
    feature = {
        "text": int64_feature(ids)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def main(args):
    log_level = logging._nameToLevel[args.log_level]
    logger = logging.getLogger(__name__)
    logger.setLevel(logging._nameToLevel[args.log_level])

    args.input = args.input[0]

    global _g_args
    _g_args = args

    logger.info('输入文件：%s', args.input)
    if not args.output:
        basename, _ = os.path.splitext(args.input)
        args.output = Path(f'{basename}.tfrecords')
    logger.info('输出文件：%s', args.output)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    total_lines = 0

    bot = time()

    logger.info('从文件 %s 加载 SentencePice BPE 模型', args.bpe_model)
    _g_sp.load(str(args.bpe_model))

    if not args.hide_progress:
        total_lines = sum(
            1 for _ in tqdm(
                open(args.input),
                desc=f'Counting of lines in {os.path.basename(args.input)}',
                unit='line'
            )
        )

    total_examples = 0
    with tf.io.TFRecordWriter(str(args.output)) as writer, \
            open(args.input) as fp_input, \
            Pool(processes=args.pool_processes) as pool:
        map_iterable = fp_input
        if not args.hide_progress:
            map_iterable = tqdm(map_iterable, total=total_lines, desc='Mapping  ')
        reduce_iterable = pool.imap_unordered(
            pool_func, map_iterable, chunksize=args.pool_chunk)
        if not args.hide_progress:
            reduce_iterable = tqdm(
                reduce_iterable, total=total_lines, desc='Executing')
        for result in reduce_iterable:
            if result:
                total_examples += 1
                writer.write(result)

    print(
        f'执行完毕! 耗时: {timedelta(seconds=time()-bot)}. 处理语料行数: {total_lines}. 采样数: {total_examples}')


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
