#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/8/25 9:21
    @Use     : 计算IV值
"""

import math
import json
import numpy as np
import pandas as pd
from .ExcelWriter import excelWriter


class Iv(object):
    def __init__(self, datapath, dep, var_names, outfile):
        self.datapath = datapath
        self.dep = dep
        self.var_names = var_names
        self.outfile = outfile
        self.df = pd.read_csv(self.datapath)
        self.excelwriter = excelWriter(bookpath=self.outfile, sheetname="IV")

    def sloveIv(self):
        vars_dic = dict()
        total_rows = self.df.shape[0]
        for idx, var_name in enumerate(self.var_names):
            pmax = self.df[var_name].max()
            cutpoints = list(set(self.df[var_name].quantile([0.01 * n for n in range(1, 100)]))) + [pmax + 1]
            cutpoints.sort(reverse=False)
            cnt_list = []
            for j in range(len(cutpoints) - 1):
                data = self.df.loc[np.logical_and(self.df[var_name] >= cutpoints[j],
                                                  self.df[var_name] < cutpoints[j + 1]), [var_name]]
                cnt_list.append((cutpoints[j], data.shape[0]))
            cnt_list = self.combine_box(cnt_list, total_rows)
            cutpoints = cnt_list + [pmax + 1]
            datas = self.df.groupby(self.df[self.dep]).count()
            dic = dict(datas[var_name].items())
            good, bad = dic.get(0, 1), dic.get(1, 1)
            sections = list()
            iv, chi = 0, 0
            for k in range(len(cutpoints) - 1):
                data = self.df.loc[np.logical_and(self.df[var_name] >= cutpoints[k],
                                                  self.df[var_name] < cutpoints[k + 1]), [var_name, self.dep]]
                datas = data.groupby(data[self.dep]).count()
                dic = dict(datas[var_name].items())
                dic = {str(int(k)): int(v) for (k, v) in dic.items()}
                dic["cnt"] = sum(dic.values())
                dic["bad_rate"] = dic.get('1', 0) / (dic["cnt"] + 1e-8)
                dic["range"] = "%s*%s" % (cutpoints[k], cutpoints[k + 1])
                a = dic.get('0', 1) / good
                b = dic.get('1', 1) / bad
                dic["woe"] = math.log(a / b)
                dic["iv"] = (a - b) * math.log(a / b)
                iv += dic["iv"]
                N = good + bad + 1e-10
                A = dic.get('1', 1) + 1e-10
                B = dic.get('0', 1) + 1e-10
                C = bad - A + 1e-10
                D = good - B + 1e-10
                dic["chi"] = N * (A * D - B * C) ** 2 / (A + C) / (A + B) / (B + D) / (C + D)
                chi += dic["chi"]
                sections.append(dic)
            vars_dic[(var_name, len(sections), json.dumps(sections), chi)] = iv
            print(idx, var_name, len(cutpoints))
        vars_dic2lis = sorted(vars_dic.items(), key=lambda x: x[1], reverse=True)
        self.excelwriter.writeLine(0, 0, ["变量", "IV值", "CHI值", "分箱数", "分箱信息"], bold=1)
        row_num = 1
        for (tupl, iv) in vars_dic2lis:
            var_name, box_num, dic, chi = tupl
            self.excelwriter.writeLine(row=row_num, start_col=0, values=[var_name, iv, chi, box_num, dic])
            row_num += 1

    def combine_box(self, cnt_list, total_rows):
        min_ratio = min([cnt for (cutpoint, cnt) in cnt_list]) / total_rows
        while len(cnt_list) >= 10 or min_ratio < 0.05:
            cnts = [cnt for (cutpoint, cnt) in cnt_list]
            min_index = cnts.index(min(cnts))
            if min_index == 0:
                cnt_list = [(cnt_list[0][0], cnt_list[0][1] + cnt_list[1][1])] + cnt_list[2:]
            elif min_index == len(cnt_list) - 1:
                cnt_list = cnt_list[:-2] + [(cnt_list[-2][0], cnt_list[-1][1] + cnt_list[-2][1])]
            elif cnt_list[min_index-1] < cnt_list[min_index+1]:
                cnt_list = cnt_list[:min_index - 1] + \
                           [(cnt_list[min_index - 1][0], cnt_list[min_index - 1][1] +
                             cnt_list[min_index][1])] + cnt_list[min_index + 1:]
            else:
                cnt_list = cnt_list[:min_index] + \
                           [(cnt_list[min_index][0], cnt_list[min_index][1] +
                             cnt_list[min_index+1][1])] + cnt_list[min_index+2:]
            min_ratio = min([cnt for (cutpoint, cnt) in cnt_list]) / total_rows
        cutpoints = [cutpoint for (cutpoint, cnt) in cnt_list]
        return cutpoints
