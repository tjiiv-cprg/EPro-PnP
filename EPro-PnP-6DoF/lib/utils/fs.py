"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

# File: fs.py from tensorpack

from __future__ import print_function, division, absolute_import
import os
from six.moves import urllib
import errno
import tqdm
from . import fancy_logger as logger

__all__ = ['mkdir_p', 'download', 'recursive_walk']


def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def download(url, dir, filename=None, expect_size=None):
    """
    Download URL to a directory.
    Will figure out the filename automatically from URL, if not given.
    """
    mkdir_p(dir)
    if filename is None:
        filename = url.split('/')[-1]
    fpath = os.path.join(dir, filename)

    if os.path.isfile(fpath):
        if expect_size is not None and os.stat(fpath).st_size == expect_size:
            logger.info("File {} exists! Skip download.".format(filename))
            return fpath
        else:
            logger.warn("File {} exists. Will overwrite with a new download!".format(filename))

    def hook(t):
        last_b = [0]

        def inner(b, bsize, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner
    try:
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=hook(t))
        statinfo = os.stat(fpath)
        size = statinfo.st_size
    except IOError:
        logger.error("Failed to download {}".format(url))
        raise
    assert size > 0, "Downloaded an empty file from {}!".format(url)

    if expect_size is not None and size != expect_size:
        logger.error("File downloaded from {} does not match the expected size!".format(url))
        logger.error("You may have downloaded a broken file, or the upstream may have modified the file.")

    # TODO human-readable size
    logger.info('Succesfully downloaded ' + filename + ". " + str(size) + ' bytes.')
    return fpath


def recursive_walk(rootdir):
    """
    Yields:
        str: All files in rootdir, recursively.
    """
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

