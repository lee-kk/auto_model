#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/9/17 10:27
    @Use     : 训练逻辑回归模型
"""

import math
import numpy as np
import xgboost as xgb
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from Base.ExcelWriter import excelWriter


class auto_lr(object):
    def __init__(self, datasets, uid, dep, weight, var_names, params, max_del_var_nums=0):
        self.datasets = datasets
        self.uid = uid
        self.dep = dep
        self.weight = weight
        self.var_names = var_names
        self.params = params
        self.max_del_var_nums = max_del_var_nums
        self.trainexcelwriter = excelWriter(bookpath=r"LogisticRegression_train_tmp.xlsx", sheetname="vars")
        self.ksexcelwriter = excelWriter(bookpath=r"KS.xlsx", sheetname="ks")
        self.row_num = 0
        self.col_num = 0
        self.df = self.datasets.get("dev", "")

    def training(self, modelfile="LogisticRegression_model.pkl", output_scores=list()):
        dev_data = self.datasets.get("dev", "")
        val_data = self.datasets.get("val", "")
        off_data = self.datasets.get("off", "")
        model = LogisticRegression(C=self.params.get("C", 1),
                                   max_iter=self.params.get("max_iter", 100),
                                   warm_start=self.params.get("warm_start", False),
                                   random_state=7,class_weight='balanced')
        model.fit(X=dev_data[self.var_names], y=dev_data[self.dep], sample_weight=dev_data[self.weight])
        devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        valks, valpsi, offks, offpsi = 0, 0, 0, 0
        if not isinstance(val_data, str):
            valks = self.sloveKS(model, val_data[self.var_names], val_data[self.dep], val_data[self.weight])
            valpsi = self.slovePSI(model, dev_data[self.var_names], val_data[self.var_names])
        if not isinstance(off_data, str):
            offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
            offpsi = self.slovePSI(model, dev_data[self.var_names], off_data[self.var_names])
        dic = {"devks": float(devks), "valks": float(valks), "offks": offks, "valpsi": float(valpsi), "offpsi": offpsi}
        print(dic)
        joblib.dump(model, modelfile)
        self.outputScore(model, output_scores)
        self.plotKS(model, bins=20)
        self.outputcoef(model=model)
        self.correlationVars()

    def correlationVars(self):
        excelwriter = excelWriter(u"变量相关性.xlsx", u"corr")
        row_num = 0
        corr_df = self.df[self.var_names].corr()
        excelwriter.writeLine(row_num, 0, [""] + self.var_names)
        for var_name in self.var_names:
            row_num += 1
            excelwriter.writeLine(row_num, 0, [var_name] + [cor for cor in corr_df[var_name].values])

    def outputcoef(self, model):
        excelwriter = excelWriter(u"逻辑回归.xlsx", u"报告")
        row_num = 0
        excelwriter.write(row_num, 0, u"1. 模型系数")
        row_num += 1
        excelwriter.writeLine(row_num, 0, [u"Variable", "DF", u"Estimate", "IV", "WaldChiSq", "ProbChiSq"])
        row_num += 1
        excelwriter.writeLine(row_num, 0, ["Intercept", 1, model.intercept_, "-", "-", "-"])
        for idx, varname in enumerate(self.var_names):
            row_num += 1
            Iv, WaldChiSq, ProbChiSq = self.sloveSignalIv(varname)
            excelwriter.writeLine(row_num, 0, [varname, 1, model.coef_[0][idx], Iv, WaldChiSq, ProbChiSq])

    def sloveSignalIv(self, var_name):
        pmax = self.df[var_name].max()
        cutpoints = list(set(self.df[var_name].quantile([0.01 * n for n in range(1, 100)]))) + [pmax + 1]
        cutpoints.sort(reverse=False)
        cnt_list = []
        for j in range(len(cutpoints) - 1):
            data = self.df.loc[np.logical_and(self.df[var_name] >= cutpoints[j],
                                              self.df[var_name] < cutpoints[j + 1]), [var_name]]
            cnt_list.append((cutpoints[j], data.shape[0]))
        cnt_list = self.signal_combine_box(cnt_list)
        cutpoints = cnt_list + [pmax + 1]
        datas = self.df.groupby(self.df[self.dep]).count()
        dic = dict(datas[var_name].items())
        good, bad = dic.get(0, 1), dic.get(1, 1)
        sections = list()
        Iv, WaldChiSq, ProbChiSq = 0, 0, 0
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
            Iv += dic["iv"]
            N = good + bad
            A = dic.get('1', 1)
            B = dic.get('0', 1)
            C = bad - A
            D = good - B
            dic["WaldChiSq"] = N * (A * D - B * C) ** 2 / (A + C) / (A + B) / (B + D) / (C + D)
            dic["ProbChiSq"] = (A * D - B * C) ** 2 / (A + C) / (A + B) / (B + D) / (C + D)
            WaldChiSq += dic["WaldChiSq"]
            ProbChiSq += dic["ProbChiSq"]
            sections.append(dic)
        return Iv, WaldChiSq, ProbChiSq

    def signal_combine_box(self, cnt_list):
        total_rows = self.df.shape[0]
        min_ratio = min([cnt for (cutpoint, cnt) in cnt_list]) / total_rows
        while len(cnt_list) >= 8 or min_ratio < 0.01:
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

    def outputScore(self, model, output_scores):
        for output_score in output_scores:
            tdf = self.datasets.get(output_score, "")
            if not isinstance(tdf, str):
                f = open("data\%s_score.txt" % output_score, "w")
                f.write("%s\t%s\tscore\n" % (self.uid, self.dep))
                UID, X, Y = tdf[self.uid], tdf[self.var_names], tdf[self.dep]
                Result = [s[1] for s in model.predict_proba(X)]
                for i in range(tdf.shape[0]):
                    f.write("%s\t%s\t%s\n" % (UID[i], Y[i], Result[i]))
                f.close()

    def sloveKS(self, model, X, Y, Weight):
        Y_predict = [s[1] for s in model.predict_proba(X)]
        nrows = X.shape[0]
        lis = [(Y_predict[i], Y.values[i], Weight[i]) for i in range(nrows)]
        ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)
        KS = list()
        bad = sum([w for (p, y, w) in ks_lis if y > 0.5])
        good = sum([w for (p, y, w) in ks_lis if y <= 0.5])
        bad_cnt, good_cnt = 0, 0
        for (p, y, w) in ks_lis:
            if y > 0.5:
                bad_cnt += w
            else:
                good_cnt += w
            ks = math.fabs((bad_cnt/bad)-(good_cnt/good))
            KS.append(ks)
        return max(KS)

    def slovePSI(self, model, dev_x, val_x):
        dev_predict_y = [s[1] for s in model.predict_proba(dev_x)]
        dev_nrows = dev_x.shape[0]
        dev_predict_y.sort()
        cutpoint = [-100] + [dev_predict_y[int(dev_nrows/10*i)] for i in range(1, 10)] + [100]
        cutpoint = list(set(cutpoint))
        cutpoint.sort()
        val_predict_y = [s[1] for s in list(model.predict_proba(val_x))]
        val_nrows = val_x.shape[0]
        PSI = 0
        for i in range(len(cutpoint)-1):
            start_point, end_point = cutpoint[i], cutpoint[i+1]
            dev_cnt = [p for p in dev_predict_y if start_point <= p < end_point]
            dev_ratio = len(dev_cnt) / dev_nrows + 1e-10
            val_cnt = [p for p in val_predict_y if start_point <= p < end_point]
            val_ratio = len(val_cnt) / val_nrows + 1e-10
            psi = (dev_ratio - val_ratio) * math.log(dev_ratio/val_ratio)
            PSI += psi
        return PSI

    def writeVarImportance(self, scores, dic):
        self.row_num = 0
        for (name, value) in dic.items():
            self.trainexcelwriter.writeLine(self.row_num, self.col_num, [name, value])
            self.row_num += 1
        self.trainexcelwriter.writeLine(self.row_num, self.col_num, ["变量", "Xgboost重要性"])
        self.row_num += 1
        dic = dict()
        for idx, col in enumerate(self.var_names):
            dic[col] = np.float32(scores[idx]).item()
        doc = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        for (var_name, sc) in doc:
            self.trainexcelwriter.writeLine(self.row_num, self.col_num, [var_name, sc])
            self.row_num += 1
        self.col_num += 2

    def plotKS(self, model, bins=10):
        self.row_num, self.col_num = 0, 0
        for (dataname, datavalue) in self.datasets.items():
            if not isinstance(datavalue, str):
                Y_predict = [s[1] for s in model.predict_proba(datavalue[self.var_names])]
                Y = datavalue[self.dep]
                Weight = datavalue[self.weight]
                nrows = Y.shape[0]
                lis = [(Y_predict[i], Y[i], Weight[i]) for i in range(nrows)]
                ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)
                bin_num = int(nrows/bins+1)
                bad = sum([w for (p, y, w) in ks_lis if y > 0.5])
                good = sum([w for (p, y, w) in ks_lis if y <= 0.5])
                bad_cnt, good_cnt = 0, 0
                KS = []
                self.ksexcelwriter.write(self.row_num, self.col_num, "%s ks plot" % dataname)
                self.row_num += 1
                lis = ["Rank", "#Total", "Max score", "Min score", "# Bad", "# Good",
                       "Cum % Total Bad", "Cum % Total Good", "K-S"]
                self.ksexcelwriter.writeLine(self.row_num, 0, lis)
                self.row_num += 1
                self.ksexcelwriter.writeLine(self.row_num, self.col_num, [0 for i in range(9)])
                self.row_num += 1
                for j in range(bins):
                    ds = ks_lis[j*bin_num: min((j+1)*bin_num, nrows)]
                    bad1 = sum([w for (p, y, w) in ds if y > 0.5])
                    good1 = sum([w for (p, y, w) in ds if y <= 0.5])
                    bad_cnt += bad1
                    good_cnt += good1
                    ks = math.fabs((bad_cnt / bad) - (good_cnt / good))
                    KS.append(ks)
                    lis = [j+1, bad1+good1, float(ds[0][0]), float(ds[-1][0]), bad1, good1,
                           bad_cnt / bad, good_cnt / good, ks]
                    self.ksexcelwriter.writeLine(self.row_num, self.col_num, lis)
                    self.row_num += 1
                lis = ["Total", bad+good, float(ks_lis[0][0]), float(ks_lis[-1][0]), int(bad), int(good),
                       np.float(bad_cnt / bad), np.float(good_cnt / good), max(KS)]
                self.ksexcelwriter.writeLine(self.row_num, self.col_num, lis)
                print("%s ks: %s" % (dataname, max(KS)))
                self.row_num += 3

    def params_choose(self):
        dev_data = self.datasets.get("dev", "")
        off_data = self.datasets.get("off", "")
        params = {
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "min_child_weight": 1,
            "subsample": 1,
            "scale_pos_weight": 1
        }
        model = xgb.XGBClassifier(list(params.items()), random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
        train_number = 0
        best_param = {"devks": devks, "offks": offks}
        print("train_number: %s, best_param: %s, params: %s" % (train_number, best_param, params))
        for key, value in params.items():
            if not isinstance(value, list):
                best_param[key] = value
                continue
            for val in value:
                tmp = deepcopy(best_param)
                model = xgb.XGBClassifier(learning_rate=tmp.get("learning_rate", 0.1),
                                          n_estimators=tmp.get("n_estimators", 100),
                                          max_depth=tmp.get("max_depth", 3),
                                          min_child_weight=tmp.get("min_child_weight", 1),
                                          subsample=tmp.get("subsample", 1),
                                          objective=tmp.get("objective", "binary:logistic"),
                                          nthread=tmp.get("nthread", 8),
                                          scale_pos_weight=tmp.get("scale_pos_weight", 1),
                                          random_state=tmp.get("random_state", 7),
                                          n_jobs=tmp.get("n_jobs", 8))
                model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
                offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
                if offks > best_param.get("offks", 0):
                    best_param[key] = val
                    devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
                    best_param["devks"], best_param["offks"] = devks, offks
                print("%s = %s, best_param: %s, offks: %s" % (key, val, best_param, offks))

    def auto_choose_params(self):
        dev_data = self.datasets.get("dev", "")
        off_data = self.datasets.get("off", "")
        params = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_weight": 1,
            "subsample": 1,
            "scale_pos_weight": 1
        }
        model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                  learning_rate=params.get("learning_rate", 0.05),
                                  n_estimators=params.get("n_estimators", 100),
                                  min_child_weight=params.get("min_child_weight", 1),
                                  subsample=params.get("subsample", 1),
                                  scale_pos_weight=params.get("scale_pos_weight", 1),
                                  nthread=8,
                                  n_jobs=8,
                                  random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
        train_number = 0
        print("train_number: %s, offks: %s, params: %s" % (train_number, offks, params))
        while True:
            dic = {
                "max_depth": [2, 1, -1, -2],
                "learning_rate": [0.1, 0.001, -0.001, -0.1],
                "n_estimators": [20, 1, -1, -20],
                "min_child_weight": [20, 1, -1, -20],
                "subsample": [0.1, 0.001, -0.001, -0.1],
                "scale_pos_weight": [20, 1, -1, -20]
            }
            outks = []
            for (key, values) in dic.items():
                for v in values:
                    if v + params[key] > 0:
                        params, offks, train_number = self.check_params(dev_data, off_data, params,
                                                                        key, train_number, v, offks)
                        outks.append(offks)
            if len(set(outks)) == 1:
                break
        print("Best params: ", params)
        model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                  learning_rate=params.get("learning_rate", 0.05),
                                  n_estimators=params.get("n_estimators", 100),
                                  min_child_weight=params.get("min_child_weight", 1),
                                  subsample=params.get("subsample", 1),
                                  scale_pos_weight=params.get("scale_pos_weight", 1),
                                  nthread=8,
                                  n_jobs=8,
                                  random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        self.plotKS(model=model, bins=20)

    def check_params(self, dev_data, off_data, params, param, train_number, step, offks):
        oldks = offks
        while True:
            try:
                params[param] += step
                model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                          learning_rate=params.get("learning_rate", 0.05),
                                          n_estimators=params.get("n_estimators", 100),
                                          min_child_weight=params.get("min_child_weight", 1),
                                          subsample=params.get("subsample", 1),
                                          scale_pos_weight=params.get("scale_pos_weight", 1),
                                          nthread=10,
                                          n_jobs=10,
                                          random_state=7)
                model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
                offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
                train_number += 1
                if offks < oldks:
                    print("(Bad) train_number：%s,  offks: %s, params: %s" % (train_number, offks, params))
                    break
                print("(Good) train_number：%s,  offks: %s, params: %s" % (train_number, offks, params))
                oldks = offks
            except:
                break
        params[param] -= step
        return params, oldks, train_number

    def auto_delete_vars(self):
        dev_data = self.datasets.get("dev", "")
        off_data = self.datasets.get("off", "")
        params = self.params
        model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                  learning_rate=params.get("learning_rate", 0.05),
                                  n_estimators=params.get("n_estimators", 100),
                                  min_child_weight=params.get("min_child_weight", 1),
                                  subsample=params.get("subsample", 1),
                                  scale_pos_weight=params.get("scale_pos_weight", 1),
                                  reg_lambda=params.get("reg_lambda", 1),
                                  nthread=8,
                                  n_jobs=8,
                                  random_state=7)
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
        train_number = 0
        print("train_number: %s, offks: %s" % (train_number, offks))
        del_list = list()
        oldks = offks
        while True:
            flag = True
            for var_name in self.var_names:
                model = xgb.XGBClassifier(max_depth=params.get("max_depth", 3),
                                          learning_rate=params.get("learning_rate", 0.05),
                                          n_estimators=params.get("n_estimators", 100),
                                          min_child_weight=params.get("min_child_weight", 1),
                                          subsample=params.get("subsample", 1),
                                          scale_pos_weight=params.get("scale_pos_weight", 1),
                                          reg_lambda=params.get("reg_lambda", 1),
                                          nthread=10,
                                          n_jobs=10,
                                          random_state=7)
                names = [var for var in self.var_names if var_name != var]
                model.fit(dev_data[names], dev_data[self.dep], dev_data[self.weight])
                train_number += 1
                offks = self.sloveKS(model, off_data[names], off_data[self.dep], off_data[self.weight])
                if offks >= oldks:
                    oldks = offks
                    flag = False
                    del_list.append(var_name)
                    print("(Good) train_n: %s, offks: %s by vars: %s" % (train_number, offks, var_name))
                    self.var_names = names
                else:
                    print("(Bad) train_n: %s, offks: %s by vars: %s" % (train_number, offks, len(self.var_names)))
            if flag:
                break
        print("(End) train_n: %s, offks: %s del_list_vars: %s" % (train_number, offks, del_list))
