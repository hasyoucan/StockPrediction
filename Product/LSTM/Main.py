#! /usr/bin/env python3

# coding: utf-8

import argparse

from Predict import Predict
from TestValidate import TestValidate


class Main:

    def run(self):

        self.__get_opt()

        if self.do_prediction:
            print('Do prediction.')
            predict = Predict()
            predict.set_draw_graph(self.draw_graph)
            predict.predict(self.stock_data_files,
                            self.target_stock, self.date_file)
        else:
            print('Do test and validation.')
            tv = TestValidate()
            tv.set_draw_graph(self.draw_graph)
            tv.test_predict(self.stock_data_files,
                            self.target_stock, self.date_file)

    def __get_opt(self):
        try:
            parser = argparse.ArgumentParser(
                description='Executes testing and validation for stock price prediction.')
            parser.add_argument('-p', required=False, action='store_true', dest='do_prediction',
                                help='do prediction')
            parser.add_argument('-d', required=False, dest='date_file', default=',date.txt',
                                help='date file')
            parser.add_argument('-t', required=True, dest='target_stock',
                                help='target stock')
            parser.add_argument('-g', required=False, action='store_true', dest='draw_graph',
                                help='draws loss and accracy graph')
            parser.add_argument('stock_data_files', nargs='+', metavar='FILE',
                                help='stock data files')
            args = parser.parse_args()

            self.do_prediction = args.do_prediction
            self.date_file = args.date_file
            self.target_stock = args.target_stock
            self.draw_graph = args.draw_graph
            self.stock_data_files = args.stock_data_files

        except Exception as e:
            print(e)
            raise e



###############################################################################
if __name__ == '__main__':
    main = Main()
    main.run()
