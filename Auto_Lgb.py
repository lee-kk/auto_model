#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  :KK
    @Time    : 2020/9/13 10:54
    @Use     : 训练LightGbm模型
    @Methods :
    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "auc"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 1
    }
    lightGbm(datasets=datasets, uid=uid, dep=dep, weight=weight, var_names=var_names, params=params).\
        training(modelfile="lightGbm_model.pkl", output_scores=[])
"""

import math
import numpy as np
import lightgbm as lgb
from sklearn.externals import joblib
from Base.ExcelWriter import excelWriter


class auto_lgb(object):
    def __init__(self, datasets, uid, dep, weight, var_names, params):
        self.datasets = datasets
        self.uid = uid
        self.dep = dep
        self.weight = weight
        self.var_names = var_names
        self.params = params
        self.trainexcelwriter = excelWriter(bookpath=r"lightgbm_train_tmp.xlsx", sheetname="vars")
        self.ksexcelwriter = excelWriter(bookpath=r"lightgbm_KS.xlsx", sheetname="ks")
        self.row_num = 0
        self.col_num = 0

    def training(self, modelfile="lightgbm_model_pkl", output_scores=list()):
        dev_data = self.datasets.get("dev", "")
        val_data = self.datasets.get("val", "")
        off_data = self.datasets.get("off", "")
        # model = lgb.LGBMClassifier(task="train", objective="regression", learning_rate=0.5)
        # model.fit(X=dev_data[self.var_names], y=dev_data[self.dep], sample_weight=dev_data[self.weight])
        model = lgb.train(params=self.params, train_set=lgb.Dataset(dev_data[self.var_names], dev_data[self.dep]),
                          num_boost_round=20, valid_sets=lgb.Dataset(val_data[self.var_names], val_data[self.dep]),
                          early_stopping_rounds=5)
        devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
        valks = self.sloveKS(model, val_data[self.var_names], val_data[self.dep], val_data[self.weight])
        valpsi = self.slovePSI(model, dev_data[self.var_names], val_data[self.var_names])
        offks = self.sloveKS(model, off_data[self.var_names], off_data[self.dep], off_data[self.weight])
        offpsi = self.slovePSI(model, dev_data[self.var_names], off_data[self.var_names])
        dic = {"devks": float(devks), "valks": float(valks), "offks": offks, "valpsi": float(valpsi), "offpsi": offpsi}
        print("ks: ", dic)
        joblib.dump(model, modelfile)
        self.outputScore(model, output_scores)
        self.plotKS(model, bins=20)

    def outputScore(self, model, output_scores):
        for output_score in output_scores:
            tdf = self.datasets.get(output_score, "")
            if not isinstance(tdf, str):
                f = open("data\%s_score.txt" % output_score, "w")
                f.write("%s\t%s\tscore\n" % (self.uid, self.dep))
                UID, X, Y = tdf[self.uid], tdf[self.var_names], tdf[self.dep]
                Result = model.predict(X)
                for i in range(tdf.shape[0]):
                    f.write("%s\t%s\t%s\n" % (UID[i], Y[i], Result[i]))
                f.close()

    def sloveKS(self, model, X, Y, Weight):
        Y_predict = model.predict(X)
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
        dev_predict_y = model.predict(dev_x)
        dev_nrows = dev_x.shape[0]
        dev_predict_y.sort()
        cutpoint = [-100] + [dev_predict_y[int(dev_nrows/10*i)] for i in range(1, 10)] + [100]
        cutpoint = list(set(cutpoint))
        val_predict_y = model.predict(val_x)
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

    def plotKS(self, model, bins=10):
        self.row_num, self.col_num = 0, 0
        for (dataname, datavalue) in self.datasets.items():
            if not isinstance(datavalue, str):
                Y_predict = model.predict(datavalue[self.var_names])
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
                    lis = [j+1, len(ds), int(ds[0][0]), int(ds[-1][0]), bad1, good1,
                           bad_cnt / bad, good_cnt / good, ks]
                    self.ksexcelwriter.writeLine(self.row_num, self.col_num, lis)
                    self.row_num += 1
                lis = ["Total", len(ks_lis), int(ks_lis[0][0]), int(ks_lis[-1][0]), int(bad), int(good),
                       np.float(bad_cnt / bad), np.float(good_cnt / good), max(KS)]
                self.ksexcelwriter.writeLine(self.row_num, self.col_num, lis)
                print("%s ks: %s" % (dataname, max(KS)))
                self.row_num += 3
