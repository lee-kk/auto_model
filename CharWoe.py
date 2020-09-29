#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    # @Time    : 2020/8/26 10:54
    # @Author  : KK
    # @File    : CharWoe.py
"""
import math
from .ExcelWriter import excelWriter

#字符型特征映射为WOE编码
class charWoe(object):
    def __init__(self, datasets, keep_vars, dep, weight, vars):
        self.datasets = datasets #数据集字典，形式如{'dev':训练集,'val':测试集,'off':跨时间验证集}
        self.devf = datasets.get("dev", "") #训练集
        self.valf = datasets.get("val", "") #测试集
        self.offf = datasets.get("off", "") #跨时间验证集
        self.keep_vars = keep_vars #不参与训练的列名，通常为用户唯一标识，样本标签等信息
        self.dep = dep #标签
        self.weight = weight #样本权重
        self.vars = vars #参与建模的特征名
        self.nrows, self.ncols = self.devf.shape #样本数，特征数

    def char_woe(self):
        #得到每一类样本的个数，且加入平滑项使得bad和good都不为0
        dic = dict(self.devf.groupby([self.dep]).size())
        good, bad = dic.get(0, 0) + 1e-10, dic.get(1, 0) + 1e-10
        #在excel中记录。首先写sheet页名和标题
        excelwriter = excelWriter(bookpath="report\CharWoe.xlsx", sheetname="WOE")
        excelwriter.writeLine(0, 0, ["变量名称", "变量值", "Good", "Bad", "变量woe名称", "WOE值"], bold=1)
        excel_rows = 1
        #对每一个特征进行遍历。
        for col in self.vars:
            #得到每一个特征值对应的样本数。
            data = dict(self.devf[[col, self.dep]].groupby([col, self.dep]).size())
            '''
            当前特征取值超过100个的时候，跳过当前取值。
            因为取值过多时，WOE分箱的效率较低，建议对特征进行截断。
            出现频率过低的特征值统一赋值，放入同一箱内。
            '''
            if len(data) > 100:
                print(col, "该变量中，值的类型超过100个，建议先做截断处理...")
                continue
            #打印取值个数
            print(col,'特征中取值的个数：', len(data))
            dic = dict()
            #k是特征名和特征取值的组合，v是样本数
            for (k, v) in data.items():
                #value为特征名，dp为特征取值
                value, dp = k
                #如果找不到key设置为一个空字典
                dic.setdefault(value, {}) 
                #字典中嵌套字典
                dic[value][int(dp)] = v
            for (k, v) in dic.items():
                dic[k] = {str(int(k1)): v1 for (k1, v1) in v.items()}
                dic[k]["cnt"] = sum(v.values())
                bad_rate = round(v.get("1", 0) / dic[k]["cnt"], 6)
                
                dic[k]["bad_rate"] = bad_rate
            dic = self.combine_box_char(dic)
            for (k, v) in dic.items():
                a = v.get("0", 1) / good + 1e-10
                b = v.get("1", 1) / bad + 1e-10
                dic[k]["Good"] = v.get("0", 0)
                dic[k]["Bad"] = v.get("1", 0)
                dic[k]["woe"] = round(math.log(a / b), 6)
                dic[k]["iv"] = round((a - b) * math.log(a / b), 6)
            for (klis, v) in dic.items():
                for k in klis.split(","):
                    self.devf.loc[self.devf[col] == k, "%s_woe" % col] = v["woe"]
                    if not isinstance(self.valf, str):
                        self.valf.loc[self.valf[col] == k, "%s_woe" % col] = v["woe"]
                    if not isinstance(self.offf, str):
                        self.offf.loc[self.offf[col] == k, "%s_woe" % col] = v["woe"]
                excelwriter.writeLine(excel_rows, 0, [col, klis, v["Good"], v["Bad"], "%s_woe" % col, v["woe"]])
                excel_rows += 1
        return {"dev": self.devf, "val": self.valf, "off": self.offf}


    def combine_box_char(self, dic):
        while len(dic) >= 10:
            bad_rate_dic = {k: v["bad_rate"] for (k, v) in dic.items()}
            bad_rate_sorted = sorted(bad_rate_dic.items(), key=lambda x: x[1])
            bad_rate = [bad_rate_sorted[i + 1][1] - bad_rate_sorted[i][1] for i in range(len(bad_rate_sorted) - 1)]
            min_rate_index = bad_rate.index(min(bad_rate))
            k1, k2 = bad_rate_sorted[min_rate_index][0], bad_rate_sorted[min_rate_index + 1][0]
            dic["%s,%s" % (k1, k2)] = dict()
            dic["%s,%s" % (k1, k2)]["0"] = dic[k1].get("0", 0) + dic[k2].get("0", 0)
            dic["%s,%s" % (k1, k2)]["1"] = dic[k1].get("1", 0) + dic[k2].get("1", 0)
            dic["%s,%s" % (k1, k2)]["cnt"] = dic[k1]["cnt"] + dic[k2]["cnt"]
            dic["%s,%s" % (k1, k2)]["bad_rate"] = round(dic["%s,%s" % (k1, k2)]["1"] / dic["%s,%s" % (k1, k2)]["cnt"],
                                                        6)
            del dic[k1], dic[k2]
        min_cnt = min([v["cnt"] for v in dic.values()])
        
        while min_cnt < self.nrows * 0.05 and len(dic) > 4:
            min_key = [k for (k, v) in dic.items() if v["cnt"] == min_cnt][0]
            bad_rate_dic = {k: v["bad_rate"] for (k, v) in dic.items()}
            bad_rate_sorted = sorted(bad_rate_dic.items(), key=lambda x: x[1])
            keys = [k[0] for k in bad_rate_sorted]
            min_index = keys.index(min_key)
            if min_index == 0:
                k1, k2 = keys[:2]
            elif min_index == len(dic) - 1:
                k1, k2 = keys[-2:]
            else:
                bef_bad_rate = dic[min_key]["bad_rate"] - dic[keys[min_index - 1]]["bad_rate"]
                aft_bad_rate = dic[keys[min_index + 1]]["bad_rate"] - dic[min_key]["bad_rate"]
                if bef_bad_rate < aft_bad_rate:
                    k1, k2 = keys[min_index - 1], min_key
                else:
                    k1, k2 = min_key, keys[min_index + 1]
            dic["%s,%s" % (k1, k2)] = dict()
            dic["%s,%s" % (k1, k2)]["0"] = dic[k1].get("0", 0) + dic[k2].get("0", 0)
            dic["%s,%s" % (k1, k2)]["1"] = dic[k1].get("1", 0) + dic[k2].get("1", 0)
            dic["%s,%s" % (k1, k2)]["cnt"] = dic[k1]["cnt"] + dic[k2]["cnt"]
            dic["%s,%s" % (k1, k2)]["bad_rate"] = round(dic["%s,%s" % (k1, k2)]["1"] / dic["%s,%s" % (k1, k2)]["cnt"],
                                                        6)
            del dic[k1], dic[k2]
            min_cnt = min([v["cnt"] for v in dic.values()])
        return dic






