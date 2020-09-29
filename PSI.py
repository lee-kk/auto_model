#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/7/22 14:31
    @Use     : 计算变量的PSI
"""

import os
import numpy as np
import math
from .ExcelReader import excelReader
from .ExcelWriter import excelWriter


class Psi(object):
    def __init__(self, datasets, dep, var_names, ivfile, outfile):
        self.datasets = datasets
        self.dep = dep
        self.var_names = var_names
        self.excelreader = excelReader(file=ivfile, sheet="IV", var_names=var_names)
        self.excelwriter = excelWriter(bookpath=outfile, sheetname="PSI")
        self.row_num, self.col_num = 0, 0

    def slove(self):
        dic = self.excelreader.get_vars_by_model_trace()
        self.excelwriter.writeLine(self.row_num, 0, ["var_name", "PSI(dev-val)", "PSI(dev-off)"], bold=1)
        self.row_num += 1
        for idx, var_name in enumerate(self.var_names):
            print(idx, var_name)
            df_out = dict()
            iv, cut_json = dic.get(var_name, (0, []))
            psi_dic = dict()
            for cjson in cut_json:
                start_v, end_v = [float(v) for v in cjson.get("range", "*").split("*")]
                for dataname, df_data in self.datasets.items():
                    df_out.setdefault(dataname, dict())
                    if not isinstance(df_data, str):
                        data = df_data.loc[np.logical_and(df_data[var_name] >= start_v, df_data[var_name] < end_v),
                                            [var_name, self.dep]]
                        datas = data.groupby(data[self.dep]).count()
                        doc = dict(datas[var_name].items())
                        cnt = sum(doc.values()) + 1e-10
                        dev_bad_rate = doc.get(1, 0) / cnt
                        df_out[dataname][start_v] = [cnt, math.log(dev_bad_rate+1e-10)]
                        psi_dic.setdefault(dataname, [])
                        psi_dic[dataname].append(cnt)
            valpsi = self.var_PSI(psi_dic.get("dev", []), psi_dic.get("val", [])) if "val" in psi_dic else "-"
            offpsi = self.var_PSI(psi_dic.get("dev", []), psi_dic.get("off", [])) if "off" in psi_dic else "-"
            self.excelwriter.writeLine(self.row_num, start_col=0, values=[var_name, valpsi, offpsi])
            self.row_num += 1

    def var_PSI(self, dev_data, val_data):
        dev_cnt, val_cnt = sum(dev_data), sum(val_data)
        if dev_cnt * val_cnt == 0:
            return None
        PSI = 0
        for i in range(len(dev_data)):
            dev_ratio = dev_data[i] / dev_cnt
            val_ratio = val_data[i] / val_cnt + 1e-10
            psi = (dev_ratio - val_ratio) * math.log(dev_ratio/val_ratio)
            PSI += psi
        return PSI
