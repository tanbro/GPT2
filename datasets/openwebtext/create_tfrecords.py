import argparse
import glob
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import ftfy
import numpy as np
from tqdm.auto import tqdm

import tensorflow as tf

import encoder


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_chunks_lazy(iterable, size):
    """将可迭代对象分块

    返回的可迭代对象，每次迭代获得分好块的元素的列表

    Parameters
    ----------

    iterable :
        可迭代对象

    size : int
        每块的大小
    """
    snippet = []
    for i in iterable:
        snippet.append(i)
        if len(snippet) < size:
            continue
        yield snippet
        snippet = []
    if snippet:
        yield snippet


def make_chunks(iterable, size):
    return list(make_chunks_lazy(iterable, size))


enc = None


def pool_init(args):
    global enc
    enc = encoder.get_spm_bpe_encoder(args.bpe_model)


def create_file(args, data):
    output_dir = args.output_dir
    output_name = args.output_name
    encoder_path = args.bpe_model
    logs_dir = args.logs_dir
    #
    index, files = data
    #
    s = '{}_{}.tfrecords'.format(output_name, index)
    # Hack-y, if file of same name is in log dir, sign that the file is complete, so skip
    if os.path.exists(os.path.join(logs_dir, s)):
        print(
            f'file of same name "{s}" is in log dir "{logs_dir}" !', file=sys.stderr)
        return
    if os.path.exists(os.path.join(output_dir, s)):  # Unfinished file, remove
        os.remove(os.path.join(output_dir, s))

    with tf.io.TFRecordWriter(os.path.join(output_dir, s)) as writer:
        good_files = 0
        current = None
        for fn in files:
            with tf.io.gfile.GFile(fn, "r") as f:
                d = f.read()
            d = ftfy.fix_text(d, normalization='NFKC')
            data = np.array(enc.encode(d), np.int32)
            # If text is shorter than 25 tokens, or all tokens are 0, ignore
            if data.shape[0] < 25 or (data == 0).all():
                continue
            hash = fn.split("/")[-1].split(".")[0]
            feature = {
                "hash": _bytes_feature(hash.encode()),
                "text": _int64_feature(data)
            }
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            good_files += 1
    # File complete
    # Create mark that file is finished in logdir
    with open(os.path.join(logs_dir, s), "w") as f:
        f.write("{} / {}".format(good_files, len(files)))
    with open(os.path.join(logs_dir, "good_files.log"), "a") as f:
        f.write("{}: {} / {}".format(index, good_files, len(files)))

    return good_files


def set_get_args():
    parser = argparse.ArgumentParser(
        description='用 BPE 模型将一批文本文件转为 *.tfrecord 文件'
    )
    parser.add_argument('-i', '--input-pat', type=str,
                        required=True, help='输入目录文件，即我们的 .txt 文件的表达式，如 /a/b/c/*.txt')
    parser.add_argument('-o', '--output-dir', type=Path,
                        required=True, help='输出目录')
    parser.add_argument('-p', '--input-per-output', type=int,
                        default=1, help='多少个输入文件对应于一个输出文件 (default=%(default)s)')
    parser.add_argument('-n', '--output-name', type=Path,
                        required=True, help='输出文件名')
    parser.add_argument('-l', '--logs-dir', type=Path,
                        default='logs', help='日志输出目录 (default=%(default)s)')
    parser.add_argument('-s', '--processes', type=int, help='进程数')
    parser.add_argument('-m', '--bpe-model', type=str,
                        required=True, help='BPE 模型文件的路径')
    arguments = parser.parse_args()
    return arguments


def main(args):
    # params
    input_pat = args.input_pat
    output_dir = args.output_dir
    files_per_chunk = args.input_per_output
    processes = args.processes
    logs_dir = args.logs_dir
    #
    print(f'输出目录: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'日志目录: {output_dir}')
    os.makedirs(logs_dir, exist_ok=True)
    print(f'匹配文件: {input_pat} ...')
    files = glob.glob(input_pat, recursive=True)
    file_chunks = make_chunks(files, files_per_chunk)
    print("Got {} files, divided into {} chunks.".format(
        len(files), len(file_chunks)))

    start = time.time()
    pool = Pool(processes=processes, initializer=pool_init, initargs=(args,))
    good = 0
    for g in tqdm(
        pool.imap_unordered(partial(create_file, args),
                            enumerate(file_chunks)),
        total=len(file_chunks)
    ):
        good += g
    end = time.time()

    print(
        "Done! In {:.2f}s, {} / {} good files.".format(
            end - start, str(good), len(files)
        ))


if __name__ == "__main__":
    args = set_get_args()
    main(args)
