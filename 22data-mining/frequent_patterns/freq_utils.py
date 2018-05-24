# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 


def loadData(fileName):
    f = open(fileName, 'r')
    dataSet = []
    for line in f.readlines():
        line_arr = line.strip().split(',')
        dataSet.append(line_arr)
    return dataSet


def loadDblpData(file_path, flag, row_num=1):
    '''
    加载dblp的数据
    :param file_path:
    :return:
    '''
    dataSetDict = {}
    dataSet = []
    count = 0
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count > row_num:
                break
            line = line.strip().split(flag)
            dataSet.append(line)
            dataLine = [word for word in line]
            dataSetDict[frozenset(dataLine)] = dataSetDict.get(frozenset(dataLine), 0) + 1
            count += 1
    return dataSetDict, dataSet


def loadUnixData(fileRead, fileWrite):
    '''
    获取unix命令数据
    :param fileName:
    :return:
    '''
    f = open(fileRead, 'r')
    fwrite = open(fileWrite, "w")
    dataSet = []
    for line in f.readlines():
        line_arr = line.strip().split(',')
        setList = set(line_arr)
        line = list(setList)

        temp = ""
        for item in line:
            if temp == "":
                temp = item
            else:
                temp = temp + ',' + item
        fwrite.write(temp + "\n")
    return dataSet


def getAuthorsData(fileRead, fileWrite):
    '''
    加载原始作者数据预处理
    :param fileName:
    :return:
    '''
    f = open(fileRead, 'r')
    fwrite = open(fileWrite, "w")
    dataSet = []
    i = 0
    for line in f.readlines():
        if line == "\n":
            continue
        line = line[:len(line) - 2]
        line_arr = line.strip().split(',')
        dataSet.append(line_arr)
        fwrite.write(line + "\n")
    return dataSet


def getUnixData(fileRead, fileWrite):
    '''
    加载数据Unix用户命令数据
    :param fileName:
    :return:
    '''
    f = open(fileRead, 'r')
    fwrite = open(fileWrite, "w")
    dataSet = []
    temp = ''
    for line in f.readlines():
        if line == "\n":
            continue
        line = line.split("\n")[0]
        print(line)
        if line == "**SOF**":
            temp = ''
        elif line == "**EOF**":
            if temp == "":
                continue
            fwrite.write(temp + "\n")
        else:
            if temp == "":
                temp = line
            else:
                temp = temp + ',' + line


def printDataSet(dataSet):
    for i in range(len(dataSet)):
        for j in range(len(dataSet[i])):
            print(dataSet[i][j])
