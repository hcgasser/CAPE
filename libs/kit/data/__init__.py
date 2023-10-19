import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from enum import IntEnum
import re

import numpy as np
import pandas as pd
import yaml

import kit.globals as G
from kit.path import join
from kit.utils import get_class


class Split(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    PREDICT = 3

    def __str__(self):
        return super().__str__().split(".")[-1]

    def long_name(self):
        if self == Split.TRAIN:
            return "training"
        elif self == Split.VAL:
            return "validation"
        elif self == Split.TEST:
            return "test"
        elif self == Split.PREDICT:
            return "prediction"

    @staticmethod
    def get_palette():
        """ return the seaborn color palette linked with the splits """
        return {s.long_name(): f"C{int(s)*2}" for s in Split}


class DD(OrderedDict):   
    """ implements the dot notation to access dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def from_yaml(path, doc_idx=0, eval=True, **kwargs):
        x = file_to_str(path)
        x = list(yaml.load_all(x, Loader=yaml.FullLoader))
        x = DD.from_dict(x[doc_idx])
        if eval:
            x = x.eval(**kwargs)
        return x

    @staticmethod
    def from_dict(d):
        result = DD()
        for key, value in d.items():
            if type(value) == dict:
                value = DD.from_dict(value)
            result[key] = value
        return result

    def update_from_yaml(self, path, doc_idx=0, eval=True, inplace=False, **kwargs):
        this = self if inplace else self.copy()
        x = file_to_str(path)
        x = list(yaml.load_all(x, Loader=yaml.FullLoader))
        x = DD.from_dict(x[doc_idx])
        for key, value in list(this.items()) + list(x.items()):
            this[key] = value
        if eval:
            this = this.eval(inplace=inplace, **kwargs)
        return this

    def to_dict(self):
        result = {}
        for key, value in self.items():
            if type(value) == DD:
                value = value.to_dict()
            result[key] = value
        return result

    def to_yaml(self, path, overwrite=False):
        if overwrite or not os.path.exists(path):
            this = self.make_evaluable().to_dict()
            with open(path, 'w') as outfile:
                yaml.dump(this, outfile, default_flow_style=False)

    def make_evaluable(self, inplace=False):
        this = self if inplace else DD()
        for key, value in self.items():
            if type(value) == DD:
                this[key] = value.make_evaluable()
            else:
                if key == 'CLS':
                    this[key] = f"{value.__module__}.{value.__name__}"
                elif type(value) == str:
                    this[key] = f'"{value}"'
                else:
                    this[key] = f"{value}"
        return this

    def eval(self, inplace=False, **kwargs):
        this = self if inplace else DD()
        for key, value in self.items():
            if type(value) == DD:
                this[key] = value.eval()
            else:
                if value == 'INPUT':
                    if key in G.ARGS or f"--{key}" in G.ARGS_UNKNOWN:
                        value = G.ARGS[key] if key in G.ARGS else G.ARGS_UNKNOWN[G.ARGS_UNKNOWN.index(f'--{key}')+1]
                    else:
                        print(f"{key}: ", end="")
                        value = input()
                if key == 'CLS':
                    this[key] = get_class(value)
                else:
                    this[key] = eval(value)
        return this

    def copy(self):
        return DD.from_dict(super(self.__class__, self).copy())


def join_lists(lists):
    result = []
    for list in lists:
        result += list

    return result


def list_of_lists(length):
    result = []
    for j in range(length):
        result.append([])
    return result


def get_dict_elements(dictionary, sep='-', prefix=''):
    """ returns the leaves of a nested dictionary as a list of strings
        the final level can be lists or sets. levels separated by 'sep'

    :param dictionary: a dictionary for which to return the leaves as strings
    :param sep: character separating the levels in the resulting string per leave
    :return:
    """

    if type(dictionary) == defaultdict:
        dictionary = dict(dictionary)

    result = []
    if type(dictionary) == dict:
        for key, value in dictionary.items():
            result += get_dict_elements(value, sep=sep, prefix=f"{prefix}{key}{sep}")
    else:
        if type(dictionary) == list or type(dictionary) == set:
            for leave in dictionary:
                result.append(f"{prefix}{leave}")

    return result


def get_dict_elements_by_level(dictionary, level):
    """ returns the elements on a certain level of a nested dictionary - the leaves can be lists or sets. level 0 is the highest

    :param dictionary: nested dictionary
    :param level: level to return
    :return: elements on the selected level
    """

    if level == 0:
        return list(dictionary.keys())
    else:
        result = []
        for value in dictionary.values():
            if (type(value) == list or type(value) == set) and level == 1:
                result += value
            if type(value) == dict:
                result += get_dict_elements_by_level(value, level-1)

        return result


def repair_column_names(df):
    df.columns = [c.replace(' ', '_').replace('/', '_').replace('.', '_').replace('%', 'pc').replace('(', '').
                  replace(')', '') for c in df.columns]


def read_csv(filename, **kwargs):
    df = pd.read_csv(filename, **kwargs)
    repair_column_names(df)
    return df


def str_to_file(s, filename, append=False):
    mode = 'a' if append else 'w'
    with open(filename, mode) as file:
        file.write(s)


def file_to_str(filename):
    if not os.path.exists(filename):
        return ""
    else:
        with open(filename, 'r') as file:
            return file.read()


def df_to_csv(df, filename, sep_csv="\t"):
    """ allows for the storage of dataframes with lists to CSV files

    :param df:
    :param filename:
    :param sep_csv:
    :return:
    """
    rows = []
    cols = df.columns

    columns = f"{sep_csv}".join([df.index.name] + list(cols)) + "\n"
    rows.append(columns)
    for idx, row in df.iterrows():
        zeile = []
        for entry in [row[col] for col in cols]:
            if type(entry) == str:
                entry = f"'{entry}'"
            zeile.append(str(entry))

        if type(idx) == str:
            idx = f"'{idx}'"
        rows.append(f"{str(idx)}{sep_csv}" + f"{sep_csv}".join(zeile) + "\n")

    with open(filename, "w") as file:
        file.writelines(rows)


def csv_to_df(filename, sep_csv="\t"):
    """ reads dataframes with lists from CSV files """

    rows = []
    with open(filename, "r") as file:
        rows = file.readlines()
        rows = [row[:-1] for row in rows]

    columns = rows[0].split(sep_csv)

    data = []
    for row in rows[1:]:
        zeile = [eval(entry) for entry in row.split(sep_csv)]
        data.append(zeile)

    return pd.DataFrame(columns=columns, data=data).set_index(columns[0])


def join_dfs(df1, df2, on):
    df2["a987321"] = 1
    df_result = df1.join(df2, on=on, how='left')
    missing = set(df_result.query("a987321.isnull()").reset_index()[on].values)
    df_result = df_result[df_result["a987321"] == 1].drop(labels="a987321", axis=1)
    df2.drop(labels="a987321", axis=1, inplace=True)
    return df_result, missing


def df_summary(df, filename, limit=50, add_cnt=False):
    text = ""
    for column in df.columns:
        text += f"{column}\n"
        uniques = df[column].unique()
        if len(uniques) < limit:
            for unique in uniques:
                text += f"\t{unique}"
                if add_cnt:
                    cnt = len(df.query(f'{column} == "{unique}"'))
                    text += f" ({cnt})"
                text += "\n"
        else:
            text += f"\tmore than {limit} different values"
        text += "\n\n"

    write_string_to_file(text, filename)


def df_get_last_entry(df, column):
    col = [x for x in df[column] if x != '' and pd.notnull(x)]
    if len(col) > 0:
        return col[-1]
    return None


def df_get_highest_in_category(df, tgt_column, category_column, query):
    df = df.reset_index()
    df = df.query(query)
    df = df[[tgt_column, category_column]].groupby(category_column).max()
    return df


def set_df_cell_to_np(df, idx, column, array, dtype=np.dtype('float32')):
    df.at[idx, column] = array.astype(dtype).tobytes()


def get_np_from_df_cell(df, idx, column, shape=None, dtype=np.dtype('float32')):
    value = np.frombuffer(df[idx, column], dtype=dtype)
    if shape is not None:
        value = value.reshape(shape)
    return value


def scale_df(df):
    maximum = df.max().max()
    minimum = df.min().min()

    scale = max(maximum, -minimum)

    return df/scale


def split_list(liste, fractions):
    liste = np.random.permutation(deepcopy(liste))
    result = []

    von, bis = 0, 0
    for frac in fractions:
        bis += int(len(liste) * frac)
        result.append(liste[von:bis])
        von = bis

    result.append(liste[bis:])

    return result


def write_string_to_file(s, filename):
    str_to_file(s, filename)


def read_string_from_file(filename):
    return file_to_str(filename)


def remove_lines(lines_or_file_path, regex):
    input_is_file = False
    if type(lines_or_file_path) == list:
        input_lines = lines_or_file_path
    elif type(lines_or_file_path) == str and os.path.exists(lines_or_file_path):
        input_is_file = True
        input_lines = file_to_str(lines_or_file_path).split('\n')
    else:
        raise Exception(f"lines must be a list or a file path, but is {lines_or_file_path}")
    output_lines = []

    if type(regex) == str:
        regex = [regex]

    for line in input_lines:
        if not any([re.match(r, line) is not None for r in regex]):
            output_lines.append(line)

    if input_is_file:
        str_to_file('\n'.join(output_lines), lines_or_file_path)

    return output_lines


def remove_lines_between(lines_or_file_path, re_start, re_end=None, remove_start=True, remove_end=True):
    input_is_file = False
    if type(lines_or_file_path) == list:
        input_lines = lines_or_file_path
    elif type(lines_or_file_path) == str and os.path.exists(lines_or_file_path):
        input_is_file = True
        input_lines = file_to_str(lines_or_file_path).split('\n')
    else:
        raise Exception(f"lines must be a list or a file path, but is {lines_or_file_path}")
    output_lines = []

    output, switch_next = True, False
    for line in input_lines:
        if switch_next:
            output, switch_next = not output, False

        if re_start is not None and re.match(re_start, line):
            if output:
                if remove_start:
                    output = False
                else:
                    switch_next = True

        if re_end is not None and re.match(re_end, line):
            if not output:
                if remove_end:
                    switch_next = True
                else:
                    output = True
        if output:
            output_lines.append(line)

    if input_is_file:
        str_to_file('\n'.join(output_lines), lines_or_file_path)

    return output_lines
