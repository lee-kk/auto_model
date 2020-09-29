#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/8/25 17:06
    @Use     : 绘制单变量的bivar图
"""

import math
import numpy as np
from .ExcelWriter import excelWriter
from .ExcelReader import excelReader


class bivarPlot(object):
    def __init__(self, datasets, dep, var_names, ivfile, outfile):
        self.datasets = datasets
        self.dep = dep
        self.var_names = var_names
        self.row_num = 0
        self.col_num = 0
        self.excelreader = excelReader(file=ivfile, sheet="IV", var_names=var_names)
        self.excelwriter = excelWriter(bookpath=outfile, sheetname="bivar")

    def plot(self):
        dic = self.excelreader.get_vars_by_model_trace()
        for idx, var_name in enumerate(self.var_names):
            print(idx, var_name)
            numdata = list()
            df_out = dict()
            iv, cut_json = dic.get(var_name, (0, []))
            psi_dic = dict()
            for cjson in cut_json:
                start_v, end_v = [float(v) for v in cjson.get("range", "*").split("*")]
                for dataname, data_df in self.datasets.items():
                    df_out.setdefault(dataname, dict())
                    if not isinstance(data_df, str):
                        data = data_df.loc[np.logical_and(data_df[var_name] >= start_v, data_df[var_name] < end_v),
                                            [var_name, self.dep]]
                        datas = data.groupby(data[self.dep]).count()
                        doc = dict(datas[var_name].items())
                        cnt = sum(doc.values()) + 1e-10
                        if cnt >= 0:
                            dev_bad_rate = doc.get(1, 0) / cnt
                            df_out[dataname][start_v] = [cnt, math.log(dev_bad_rate+1e-10)]
                            if dataname == "dev":
                                numdata.append(("%s~%s" % (start_v, end_v), int(cnt), math.log(dev_bad_rate+1e-10)))
                            psi_dic.setdefault(dataname, [])
                            psi_dic[dataname].append(cnt)
            self.excelwriter.write(self.row_num, 0, var_name)
            self.row_num += 1
            for name in ["val", "off"]:
                if name in psi_dic:
                    psi = self.var_PSI(psi_dic.get("dev", []), psi_dic.get(name, []))
                    self.excelwriter.writeLine(self.row_num, 0, ["PSI(dev-%s)" % name, psi])
                    self.row_num += 1
            self.row_num += 1
            self.col_num = 4
            for dataname, df_out in df_out.items():
                if df_out:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(4, 3))
                    if dataname == "dev":
                        self.excelwriter.writeLine(self.row_num, 0, ["idx", "Range", "Cnt", "Inodds"], bold=1)
                        for (idx, tpl) in enumerate(numdata):
                            Range, Cnt, Inodds = tpl
                            self.excelwriter.writeLine(self.row_num+idx+1, 0, [idx+1, Range, Cnt, Inodds])
                    y_cnt = [cnt for (cnt, rate) in df_out.values()]
                    ax1 = fig.add_subplot(111)
                    ax1.bar(range(1, 1+len(df_out)), y_cnt, color="Blue")
                    ax1.set_title(dataname)

                    y_rate = [rate for (cnt, rate) in df_out.values()]
                    ax2 = ax1.twinx()
                    ax2.plot(range(1, 1+len(df_out)), y_rate, color="Red")

                    fig.savefig("pic/%s_%s.png" % (dataname, var_name))
                    self.excelwriter.insert_image(row=self.row_num-4, col=self.col_num,
                                                  pic="pic/%s_%s.png" % (dataname, var_name))
                    plt.close()
                    self.col_num += 6
            self.row_num += 13

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
