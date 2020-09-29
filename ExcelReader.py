#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/7/24 13:30
    @Use     : 读取Excel的数据
"""

import xlrd
import json
import pandas as pd
from pandas.api.types import is_numeric_dtype
from .ExcelWriter import excelWriter


class excelReader(object):
    def __init__(self, file, sheet, var_names):
        self.book = xlrd.open_workbook(filename=file)
        self.sheet = self.book.sheet_by_name(sheet_name=sheet)
        self.nrows, self.ncols = self.sheet.nrows, self.sheet.ncols
        self.var_names = var_names

    def read(self, row, col):
        return self.sheet.cell_value(rowx=row, colx=col)

    def get_vars_by_model_trace(self):
        flag = False
        dic = dict()
        for i in range(self.nrows):
            var_name = self.sheet.cell_value(i, 0)
            if var_name == "变量":
                flag = True
                continue
            if var_name and flag and var_name in self.var_names:
                iv = self.sheet.cell_value(i, 1)
                json_cut = self.sheet.cell_value(i, 4)
                dic[var_name] = [iv, json.loads(json_cut)]
        return dic

    def getcols(self):
        var_names = []
        flag = False
        for col in range(self.ncols-1, -1, -1):
            if flag:
                break
            for row in range(self.nrows):
                var_name = self.sheet.cell_value(rowx=row, colx=col)
                if var_name == "变量":
                    flag = True
                    continue
                if var_name and flag:
                    var_names.append(var_name)
        return var_names


def getModelVars(path, dep, weight, outfile=r'model_trace.xlsx'):
    devf = pd.read_csv(path)
    excelwriter = excelWriter(bookpath=outfile, sheetname="vars")
    excelwriter.write(0, 0, "变量", bold=1)
    row_num = 1
    for col in devf.columns:
        if is_numeric_dtype(devf[col]) and col != dep and col != weight:
            excelwriter.write(row=row_num, col=0, val=col)
            row_num += 1
