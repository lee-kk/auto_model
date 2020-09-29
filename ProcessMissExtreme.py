#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/7/24 23:23
    @Use     : 极值缺失值处理
"""

import pandas as pd
from .ExcelWriter import excelWriter


class processMissExtreme(object):
    def __init__(self, devp="", valp="", offp="", var_names=[], record=r"report/DealMaxMinNull.xlsx"):
        self.devp = devp
        self.valp = valp
        self.offp = offp
        self.devf = pd.read_csv(devp) if valp != "" else ""
        self.valf = pd.read_csv(valp) if valp != "" else ""
        self.offf = pd.read_csv(offp) if offp != "" else ""
        self.var_names = var_names
        self.excelwriter = excelWriter(bookpath=record, sheetname="MaxMinNull")

    def dealing(self):
        self.excelwriter.writeLine(row=0, start_col=0, values=["var_name", "p1", "p99"], bold=1)
        for (idx, var_name) in enumerate(self.var_names):
            p01, p99 = self.devf[var_name].quantile([0.01, 0.99]).values
            p01, p99 = round(p01, 5), round(p99, 5)
            print(idx, var_name, p01, p99)
            self.excelwriter.writeLine(idx+1, 0, [var_name, p01, p99])

            self.devf.loc[self.devf[var_name].isnull(), var_name] = -100
            self.devf.loc[self.devf[var_name] < p01, var_name] = p01
            self.devf.loc[self.devf[var_name] > p99, var_name] = p99

            if not isinstance(self.valf, str):
                self.valf.loc[self.valf[var_name].isnull(), var_name] = -100
                self.valf.loc[self.valf[var_name] < p01, var_name] = p01
                self.valf.loc[self.valf[var_name] > p99, var_name] = p99

            if not isinstance(self.offf, str):
                self.offf.loc[self.offf[var_name].isnull(), var_name] = -100
                self.offf.loc[self.offf[var_name] < p01, var_name] = p01
                self.offf.loc[self.offf[var_name] > p99, var_name] = p99
        self.devf.to_csv(self.devp.replace(".csv", "_trt.csv"), index=False)
        if not isinstance(self.valf, str):
            self.valf.to_csv(self.valp.replace(".csv", "_trt.csv"), index=False)
        if not isinstance(self.offf, str):
            self.offf.to_csv(self.offp.replace(".csv", "_trt.csv"), index=False)
