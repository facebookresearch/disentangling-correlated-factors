"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from fastargs.validation import Checker


class List(Checker):

    def check(self, value):
        if isinstance(value, str):
            return eval(value)
        elif isinstance(value, list):
            return value
        raise TypeError()

    def help(self):
        return "a list or liststr"


class Dict(Checker):

    def check(self, value):
        if isinstance(value, str):
            return eval(value)
        elif isinstance(value, dict):
            return value
        raise TypeError()

    def help(self):
        return "a dict or dictstr"


class Bool(Checker):

    def check(self, value):
        value = eval(str(value))
        if not isinstance(value, bool):
            raise TypeError()
        return value

    def help(self):
        return "a boolean"


def type_select(value):
    if isinstance(value, str): return str
    elif isinstance(value, bool): return Bool()
    elif isinstance(value, int): return int
    elif isinstance(value, float): return float
    elif isinstance(value, list): return List()
    elif isinstance(value, dict): return Dict()
    else:
        from fastargs.validation import Anything
        return Anything()
