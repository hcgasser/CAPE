""" This module contains functions for working with paths. """

import os
import re
from collections import defaultdict

from kit.log import log_info


def join(*args):
    """Joins the arguments to a path and creates the directory if it does not exist."""

    if args[-1].find(".") > 0:
        folder = os.path.join(*args[:-1])
    else:
        folder = os.path.join(*args)

    if not os.path.exists(folder):
        os.makedirs(folder)
        log_info(f"created directory: {folder}")

    url = os.path.join(*args)
    return url


def get_entries(path, regex=None, returndict=True):
    """Returns a list of entires (files and sub-directories)
    in a directory, optionally filtering by a regex.

    :param path: str - path to directory
    :param regex: str - regex to match the files/sub-directories
    :param returndict: bool - if True, returns a dict of lists,
        where the keys are the filenames without prefix and suffix

    :return result: list of files or dict of lists
    """

    result = defaultdict(lambda: []) if returndict else []
    for dirname, directories, filenames in os.walk(path):
        entries = directories + filenames
        for entry in entries:
            match = re.match(regex, entry) if regex is not None else True
            if match is not None:
                l = result[entry] if returndict else result
                l.append(os.path.join(dirname, entry))

    return result


def get_max_index(path, regex):
    """Returns the highest index of an entry in a directory,
    optionally filtered by prefix and suffix.

    :param path: str - path to directory
    :param regex: str - regex to match the files/sub-directories.
        The first group must be the index.

    :return max_index: int - highest index of a file/sub-directory
    """

    max_index = 0
    for _, dirnames, filenames in os.walk(path):
        entries = dirnames + filenames
        for entry in entries:
            match = re.match(regex, entry)
            if match is not None:
                index = int(match[1])
                if index > max_index:
                    max_index = index
    return max_index
