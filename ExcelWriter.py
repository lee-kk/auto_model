#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Author  : KK
    @Time    : 2020/8/24 23:30
    @Use     : 输出Excel结果
"""

import numpy as np
import xlsxwriter


class excelWriter(object):
    def __init__(self, bookpath, sheetname):
        self.bookpath = bookpath
        self.workbook = xlsxwriter.Workbook(filename=bookpath)
        self.sheet = self.workbook.add_worksheet(name=sheetname)
        self.bold = self.workbook.add_format({'fg_color': '#87CEFA', 'font_size': 12})

    def write(self, row, col, val, bold=0):
        if bold:
            self.sheet.write(row, col, val, self.bold)
        else:
            self.sheet.write(row, col, val)

    def writeLine(self, row, start_col, values, bold=0):
        for idx, val in enumerate(values):
            self.write(row=row, col=start_col + idx, val=val, bold=bold)

    def insert_image(self, row, col, pic):
        self.sheet.insert_image(row=row, col=col, filename=pic)

    def __del__(self):
        self.workbook.close()
