#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/9/21 11:36
    @Use     : 训练Xgboost模型
"""

import math
import numpy as np
import xgboost as xgb
from copy import deepcopy
import joblib
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from scikit2pmml import scikit2pmml
from .ExcelWriter import excelWriter
from xgboost import plot_importance


class auto_xgb(object):
    def __init__(self, datasets, uid, dep, weight, var_names, params, max_del_var_nums=0):
        self.datasets = datasets
        self.uid = uid
        self.dep = dep
        self.weight = weight
        self.var_names = var_names
        self.params = params
        self.max_del_var_nums = max_del_var_nums
        self.trainexcelwriter = excelWriter(bookpath=r"report\Xgboost_train_tmp.xlsx", sheetname="vars")
        self.ksexcelwriter = excelWriter(bookpath=r"report\KS.xlsx", sheetname="ks")
        self.row_num = 0
        self.col_num = 0

    def training(self, min_score=0.0001, modelfile="", output_scores=list()):
        lis = self.var_names[:]
        dev_data = self.datasets.get("dev", "")
        val_data = self.datasets.get("val", "")
        off_data = self.datasets.get("off", "")
        model = xgb.XGBClassifier(learning_rate=self.params.get("learning_rate", 0.1),
                                  n_estimators=self.params.get("n_estimators", 100),
                                  max_depth=self.params.get("max_depth", 3),
                                  min_child_weight=self.params.get("min_child_weight", 1),
                                  subsample=self.params.get("subsample", 1),
                                  objective=self.params.get("objective", "binary:logistic"),
                                  nthread=self.params.get("nthread", 10),
                                  scale_pos_weight=self.params.get("scale_pos_weight", 1),
                                  random_state=7,
                                  n_jobs=self.params.get("n_jobs", 10),
                                  reg_lambda=self.params.get("reg_lambda", 1),
                                  missing=self.params.get("missing", None)
                                  )
        while len(lis) > 0:
            model.fit(X=dev_data[self.var_names], y=dev_data[self.dep])# , sample_weight=dev_data[self.weight])
            scores = model.feature_importances_
            lis.clear()
            for (idx, var_name) in enumerate(self.var_names):
                if scores[idx] < min_score:
                    lis.append(var_name)
                if len(lis) >= self.max_del_var_nums:
                    break
            devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
            valks, offks, valpsi, offpsi = 0.0, 0.0, 0.0, 0.0
            if not isinstance(val_data, str):
                valks = self.sloveKS(model, val_data[self.var_names], val_data[self.dep], val_data[self.weight])
                valpsi = self.slovePSI(model, dev_data[self.var_names], val_data[self.var_names])
            if not isinstance(off_data, str):
                offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
                offpsi = self.slovePSI(model, dev_data[self.var_names], off_data[self.var_names])
            dic = {"devks": float(devks), "valks": float(valks), "offks": offks,
                   "valpsi": float(valpsi), "offpsi": offpsi}
            print("del var: ", len(self.var_names), "-->", len(self.var_names) - len(lis), "ks: ", dic, ",".join(lis))
            self.writeVarImportance(scores, dic)
            self.var_names = [var_name for var_name in self.var_names if var_name not in lis]
        self.outputScore(model, output_scores)
#        plot_importance(model)
        from matplotlib import pyplot as plt
        plt.show()
        self.plotKS(model, bins=20)
        joblib.dump(model, "model.pkl")
        model = xgb.XGBClassifier(learning_rate=self.params.get("learning_rate", 0.1),
                                  n_estimators=self.params.get("n_estimators", 100),
                                  max_depth=self.params.get("max_depth", 3),
                                  min_child_weight=self.params.get("min_child_weight", 1),
                                  subsample=self.params.get("subsample", 1),
                                  objective=self.params.get("objective", "binary:logistic"),
                                  nthread=self.params.get("nthread", 10),
                                  scale_pos_weight=self.params.get("scale_pos_weight", 1),
                                  random_state=7,
                                  n_jobs=self.params.get("n_jobs", 10),
                                  reg_lambda=self.params.get("reg_lambda", 1),
                                  missing=self.params.get("missing", None)
                                  )
#        if modelfile > "":
            # scikit2pmml(estimator=model, file="model.pmml")
#            default_mapper = DataFrameMapper([(['%s' % varname], None) for varname in self.var_names])
#            pipeline = PMMLPipeline([("mapper", default_mapper), ("classifier", model)])
#            pipeline.fit(X=dev_data[self.var_names], y=dev_data[self.dep])# , weight=dev_data[self.weight])
#            sklearn2pmml(pipeline=pipeline, pmml=modelfile, with_repr=True)

    def outputScore(self, model, output_scores):
        for output_score in output_scores:
            tdf = self.datasets.get(output_score, "")
            lis = [varname for varname in self.var_names] + [self.uid, self.dep, "predict"]
            if not isinstance(tdf, str):
                excelwriter = excelWriter(bookpath=r"report\%s_score.csv" % output_score, sheetname=output_score)
                excelwriter.writeLine(0, 0, lis)
                UID, X, Y = tdf[self.uid], tdf[self.var_names], tdf[self.dep]
                Result = [s[1] for s in model.predict_proba(X)]
                for i in range(tdf.shape[0]):
                    lis = list(X.iloc[[i]].values[0]) + [UID[i], Y[i], Result[i]]
                    excelwriter.writeLine(i+1, 0, lis)
                del excelwriter

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
        self.trainexcelwriter.writeLine(self.row_num, self.col_num, ["变量", "Xgboost重要性"], bold=1)
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
                self.ksexcelwriter.write(self.row_num, self.col_num, "%s ks plot" % dataname, bold=1)
                self.row_num += 1
                lis = ["Rank", "#Total", "Max score", "Min score", "# Bad", "# Good", "Bad Rate",
                       "Cum % Total Bad", "Cum % Total Good", "K-S"]
                self.ksexcelwriter.writeLine(self.row_num, 0, lis, bold=1)
                self.row_num += 1
                self.ksexcelwriter.writeLine(self.row_num, self.col_num, [0] * 9)
                self.row_num += 1
                bad_list = list()
                bins_weight = sum(Weight) // bins + 1
                start, end = 0, 0
                for j in range(bins):
                    start = max(end, j * bin_num)
                    end = min(end + bin_num, nrows)
                    ds = ks_lis[start: end]
                    while sum([w for (p, y, w) in ds]) < bins_weight and end < nrows:
                        end = min(end+3, nrows)
                        ds = ks_lis[start: end]
                    bad1 = sum([w for (p, y, w) in ds if y > 0.5])
                    bad_list.append(bad1)
                    good1 = sum([w for (p, y, w) in ds if y <= 0.5])
                    bad_cnt += bad1
                    good_cnt += good1
                    ks = math.fabs((bad_cnt / bad) - (good_cnt / good))
                    KS.append(ks)
                    lis = [j+1, len(ds), np.float(ds[0][0]), np.float(ds[-1][0]), bad1, good1,
                           np.float(bad1/(good1+bad1)), bad_cnt / bad, good_cnt / good, ks]
                    self.ksexcelwriter.writeLine(self.row_num, self.col_num, lis)
                    self.row_num += 1
                print(bad_list)
                lis = ["Total", len(ks_lis), np.float(ks_lis[0][0]), np.float(ks_lis[-1][0]), np.float(bad),
                       np.float(good), np.float(bad/(good + bad)), np.float(bad_cnt / bad), np.float(good_cnt / good), max(KS)]
                self.ksexcelwriter.writeLine(self.row_num, self.col_num, lis)
                print("%s ks: %s" % (dataname, max(KS)))
                self.row_num += 3

    def auto_choose_params(self, target="offks"):
        """
        :param target:
                "offks": offks最大化;
                "minus": 1-abs(devks-offks) 最大化;
                "avg": (devks+offks)/2  最大化
                "weight": offks + abs(offks - devks) * 0.2 最大化
        :return: 输出最优模型变量
        """
        dev_data = self.datasets.get("dev", "")
        off_data = self.datasets.get("off", "")
        params = {
            "max_depth": 5,
            "learning_rate": 0.09,
            "n_estimators": 120,
            "min_child_weight": 50,
            "subsample": 1,
            "scale_pos_weight": 1,
            "reg_lambda": 21
        }
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
        devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
        train_number = 0
        print("train_number: %s, devks: %s, offks: %s, params: %s" % (train_number, devks, offks, params))
        dic = {
             "learning_rate": [0.05, -0.05],

             "max_depth": [1, -1],
             "n_estimators": [20, 5, -5, -20],
             "min_child_weight": [20, 5, -5, -20],
             "subsample": [0.05, -0.05],
             "scale_pos_weight": [20, 5, -5, -20],

            "reg_lambda": [10, -10]
        }
        targetks = self.target_value(old_devks=devks, old_offks=offks, target=target, devks=devks, offks=offks)
        old_devks = devks
        old_offks = offks
        while True:
            targetks_lis = []
            for (key, values) in dic.items():
                for v in values:
                    if v + params[key] > 0:
                        params, targetks, train_number = self.check_params(dev_data, off_data, params, key, train_number,
                                                                           v, target, targetks, old_devks, old_offks)
                        targetks_n = self.target_value(old_devks=old_devks, old_offks=old_offks, target=target,
                                                       devks=devks, offks=offks)
                        if targetks < targetks_n:
                            old_devks = devks
                            old_offks = offks
                            targetks_lis.append(targetks)
            print("-"*50)
            if not targetks_lis:
                break
        print("Best params: ", params)
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
        self.plotKS(model=model, bins=20)

    def check_params(self, dev_data, off_data, params, param, train_number, step, target, targetks, old_devks, old_offks):
        while True:
            try:
                if params[param] + step > 0:
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
                    devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
                    offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
                    train_number += 1
                    targetks_n = self.target_value(old_devks=old_devks, old_offks=old_offks, target=target,
                                                   devks=devks, offks=offks)
                    if targetks < targetks_n:
                        print("(Good) train_number: %s, devks: %s, offks: %s, params: %s" % (
                            train_number, devks, offks, params))
                        targetks = targetks_n
                        old_devks = devks
                        old_offks = offks
                    else:
                        print("(Bad) train_number: %s, devks: %s, offks: %s, params: %s" % (
                        train_number, devks, offks, params))
                        break
                else:
                    break
            except:
                break
        params[param] -= step
        return params, targetks, train_number

    def target_value(self, old_devks, old_offks, target, devks, offks):
        if target == "offks":
            return offks
        elif target == "avg":
            return (devks + offks) / 2
        elif target == "weight":
            return offks - abs(devks - offks) * 3
        else:
            return 1-abs(devks-offks)

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
