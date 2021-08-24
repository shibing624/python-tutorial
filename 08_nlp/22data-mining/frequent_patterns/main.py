# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import time
from eclat import eclat_zc
from freq_utils import loadDblpData, load_title_data, printDataSet, save_freqItems
from apriori import apriori_zc
from fp_growth import fp_growth


def test_fp_growth(minSup, dataSetDict, dataSet):
    freqItems = fp_growth(dataSetDict, minSup)
    freqItems = sorted(freqItems.items(), key=lambda item: item[1])
    return freqItems


def test_apriori(minSup, dataSetDict, dataSet):
    freqItems = apriori_zc(dataSet, dataSetDict, minSup)
    freqItems = sorted(freqItems.items(), key=lambda item: item[1])
    return freqItems


def test_eclat(minSup, dataSetDict, dataSet):
    freqItems = eclat_zc(dataSet, minSup)
    freqItems = sorted(freqItems.items(), key=lambda item: item[1])
    return freqItems


def print_freqItems(logo, freqItems):
    print("-------------------", logo, "---------------")
    for i in range(len(freqItems)):
        print(i, freqItems[i])
    print(len(freqItems))
    print("-------------------", logo, " end ---------------")


def do_experiment_data_size():
    data_name = 'unixData8_pro.txt'
    x_name = "Data_Size"
    data_num = 980

    step = data_num / 5  # #################################################################
    all_time = []
    x_value = []
    for k in range(5):
        minSup = data_num * 0.010
        dataSetDict, dataSet = loadDblpData(("dataSet/" + data_name), ' ', data_num)
        x_value.append(data_num)  # #################################################################
        if data_num < 0:  # #################################################################
            break
        time_fp = 0
        time_et = 0
        time_ap = 0
        freqItems_fp = {}
        freqItems_eclat = {}
        freqItems_ap = {}
        for i in range(2):
            ticks0 = time.time()
            freqItems_fp = test_fp_growth(minSup, dataSetDict, dataSet)
            time_fp += time.time() - ticks0
            ticks0 = time.time()
            freqItems_eclat = test_eclat(minSup, dataSetDict, dataSet)
            time_et += time.time() - ticks0
            ticks0 = time.time()
            freqItems_ap = test_apriori(minSup, dataSetDict, dataSet)
            time_ap += time.time() - ticks0
        print("minSup :", minSup, "      data_num :", data_num, \
              "  freqItems_fp:", " freqItems_eclat:", len(freqItems_eclat), "  freqItems_ap:", len(
                freqItems_ap))
        print("fp_growth:", time_fp / 10, "       eclat:", time_et / 10, "      apriori:", time_ap / 10)
        # print_freqItems("show", freqItems_eclat)
        data_num -= step  # #################################################################
        use_time = [time_fp / 10, time_et / 10, time_ap / 10]
        all_time.append(use_time)
        # print use_time

    y_value = []
    for i in range(len(all_time[0])):
        tmp = []
        for j in range(len(all_time)):
            tmp.append(all_time[j][i])
        y_value.append(tmp)
    return x_value, y_value


def do_experiment_min_support():
    data_name = 'unixData8_pro.txt'
    x_name = "Min_Support"
    data_num = 980
    minSup = data_num / 6

    dataSetDict, dataSet = loadDblpData(("dataSet/" + data_name), ',', data_num)
    step = minSup / 5  # #################################################################
    all_time = []
    x_value = []
    for k in range(5):

        x_value.append(minSup)  # #################################################################
        if minSup < 0:  # #################################################################
            break
        time_fp = 0
        time_et = 0
        time_ap = 0
        freqItems_fp = {}
        freqItems_eclat = {}
        freqItems_ap = {}
        for i in range(10):
            ticks0 = time.time()
            freqItems_fp = test_fp_growth(minSup, dataSetDict, dataSet)
            time_fp += time.time() - ticks0
            ticks0 = time.time()
            freqItems_eclat = test_eclat(minSup, dataSetDict, dataSet)
            time_et += time.time() - ticks0
            ticks0 = time.time()
            freqItems_ap = test_apriori(minSup, dataSetDict, dataSet)
            time_ap += time.time() - ticks0
        print("minSup :", minSup, "      data_num :", data_num, \
              " freqItems_eclat:", len(freqItems_eclat))
        print("[time spend] fp_growth:", time_fp / 10, "       eclat:", time_et / 10, "      apriori:", time_ap / 10)
        # print_freqItems("show", freqItems_eclat)
        minSup -= step  # #################################################################
        use_time = [time_fp / 10, time_et / 10, time_ap / 10]
        all_time.append(use_time)
        # print use_time
    y_value = []
    for i in range(len(all_time[0])):
        tmp = []
        for j in range(len(all_time)):
            tmp.append(all_time[j][i])
        y_value.append(tmp)
    return x_value, y_value


def do_test():
    dataSetDict, dataSet = loadDblpData(("dataSet/connectPro.txt"), ',', 100)
    minSup = 101

    # for item in freq_items:
    #     print item
    # freqItems = test_fp_growth(minSup, dataSetDict, dataSet)
    # print_freqItems("show", freqItems)
    #
    freqItems = test_eclat(minSup, dataSetDict, dataSet)
    # print_freqItems("show", freqItems)
    freqItems_eclat = test_eclat(minSup, dataSetDict, dataSet)

    # freqItems_ap = test_apriori(minSup, dataSetDict, dataSet)
    # print_freqItems("show", freqItems_ap)

    print(len(freqItems_eclat))


def do_dblp_data():
    data_name = 'dblpDataAll.txt'
    x_name = "Min_Support"
    data_num = 980
    minSup = 100
    dataSetDict, dataSet = loadDblpData(("dataSet/" + data_name), ',', data_num)

    time_fp = 0
    ticks0 = time.time()
    freqItems_fp = test_eclat(minSup, dataSetDict, dataSet)
    time_fp += time.time() - ticks0
    print(time_fp)

    for item in freqItems_fp:
        print(item)


def do_title_data():
    data_name = 'title.txt'
    x_name = "Min_Support"
    data_num = 22846
    minSup = data_num / 100
    dataSetDict, dataSet = load_title_data(("dataSet/" + data_name), ',', data_num)
    printDataSet(dataSet[:10])
    time_fp = 0
    ticks0 = time.time()
    freqItems_fp = test_eclat(minSup, dataSetDict, dataSet)
    time_fp += time.time() - ticks0
    print(time_fp)

    print(freqItems_fp[:10])
    save_freqItems(freqItems_fp, "dataSet/title_out.txt")


if __name__ == '__main__':
    # x_value, y_value = do_experiment_min_support()
    # x_value, y_value = do_experiment_data_size()
    # do_test()
    #
    do_dblp_data()
    # do_title_data()
