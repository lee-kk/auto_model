#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    # @Time    : 2020/8/22 11:25
    # @Author  : KK
    # @File    : PredictScore.py
"""

import pandas as pd
import joblib
from .ExcelWriter import excelWriter


class predictScore(object):
    def __init__(self, rawdata, model, outdata, keepvar, modelvar, dep, weight):
        self.rawdata = rawdata
        self.model = model
        self.outdata = outdata
        self.keepvar = keepvar
        self.modelvar = modelvar
        self.dep = dep
        self.weight = weight

    def predict(self):
        pdf = pd.read_csv(self.rawdata, header=0)
        model = joblib.load(self.model)
        f = open(self.outdata, "w")
        f.write("%s\tscore\n" % "\t".join(self.keepvar))
        X = pdf[self.modelvar]
        Result = [s[1] for s in model.predict_proba(X)]
        KeepVar = pdf[self.keepvar]
        for i in range(pdf.shape[0]):
            keeplis = [str(s) for s in KeepVar.iloc[[i]].values[0]]
            f.write("%s\t%s\n" % ("\t".join(keeplis), Result[i]))
        f.close()
        self.plotKS(pdf, Result, dep=self.dep, weight=self.weight)

    def plotKS(self, pdf, score, dep, weight=None, bins=20):
        if weight is None:
            pdf[weight] = 1
        snrows = pdf.shape[0]
        sdic = {i: score[i] for i in range(snrows)}
        ddic = {i: pdf[dep][i] for i in range(snrows)}
        wdic = {i: pdf[weight][i] for i in range(snrows)}
        Good = sum([pdf[weight][i] for i in range(snrows) if pdf[dep][i] < 0.5])
        Bad = sum([pdf[weight][i] for i in range(snrows) if pdf[dep][i] >= 0.5])
        binsize = int(snrows / bins) + 1
        sdoc = sorted(sdic.items(), key=lambda x: x[1], reverse=True)
        total_good, total_bad = 0, 0
        KS = list()
        excelwriter = excelWriter(bookpath="future_KS.xlsx", sheetname="future ks")
        excelwriter.write(0, 0, "Future Ks Plot")
        lis = ["Rank", "#Total", "Max score", "Min score", "# Bad", "# Good",
               "Cum % Total Bad", "Cum % Total Good", "K-S"]
        excelwriter.writeLine(1, 0, lis)
        nrows = 2
        for i in range(bins):
            min_index, max_index = i * binsize, min(Good+Bad, i * binsize + binsize)
            tmp_sdoc = sdoc[min_index:max_index]
            indexes = [index for (index, score) in tmp_sdoc]
            good1 = sum([wdic[index] for index in indexes if ddic[index] < 0.5])
            bad1 = sum([wdic[index] for index in indexes if ddic[index] >= 0.5])
            total = good1 + bad1
            total_good += good1
            total_bad += bad1
            ks = abs(total_good / Good - total_bad / Bad)
            KS.append(ks)
            lis = [i+1, total, tmp_sdoc[0][1],
                   tmp_sdoc[-1][1],
                   bad1, good1, total_bad / Bad, total_good / Good, ks]
            excelwriter.writeLine(nrows, 0, lis)
            nrows += 1
        lis = ["Total", Bad+Good, sdoc[0][1], sdoc[-1][1], Bad, Good, 1, 1, max(KS)]
        excelwriter.writeLine(nrows, 0, lis)
        print("Future ks: %s" % max(KS))
