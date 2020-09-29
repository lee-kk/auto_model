#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/7/23 14:10
    @Use     : 读取csv数据集
"""

import pandas as pd


class getDataSets(object):
    def __init__(self, devp="", valp="", offp="", usecols=[]):
        self.devf = pd.read_csv(devp, usecols=usecols) if devp != "" else ""
        self.valf = pd.read_csv(valp, usecols=usecols) if valp != "" else ""
        self.offf = pd.read_csv(offp, usecols=usecols) if offp != "" else ""

    def loading(self):
        datasets = {"dev": self.devf, "val": self.valf, "off": self.offf}
        return datasets
