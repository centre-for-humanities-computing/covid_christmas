"""
"""
import os
import sys

import numpy as np


def silence(func):
    """
    func (fun): function which you desire silences

    Examples
    >>> def addn(x,n):  # function with annoying print
    ...    print(f"adding {n} to {x}")
    ...    return x + n
    >>> add_silences = silence(addn)
    >>> add_silences(3, 1)
    4
    """
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sav = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sav
        return value
    return func_wrapper


def add_def_args(func, def_args):
    """
    func (fun): function which to add defaults arguments to
    def_args (dict): default argument given as a dictionary

    Examples
    >>> def addn(x,n):
    ...    return x + n
    >>> add3 = add_def_args(addn, {'n':3})
    >>> add3(2)
    5
    """
    def func_wrapper(*args, **kwargs):
        value = func(*args, **def_args, **kwargs)
        return value
    return func_wrapper


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
