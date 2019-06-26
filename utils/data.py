from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import math

def get_local_data(MacList=[], BeginTime=None, EndTime=None, DeviceFlag='aircondition'):
    db_info = {'user': 'smallliu',
               'password': 'LX392754xing..',
               'host': '172.28.4.118',
               'database': 'AirConditionData'  # 这里我们事先指定了数据库，后续操作只需要表即可
               }
    tableNameDic = {'innerstatus': 'innerstatus',
                    'outerstatus': 'outerstatus',
                    'innerunit': 'innerunit',
                    'outerunit': 'outerunit',
                    'ctlresponse': 'ctlresponse'
                    }
    sqlStrDic = {}
    engine = create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s/%(database)s?charset=utf8' % db_info,
                           encoding='utf-8')
    allMacFlag = 0
    sqlStr = 'select min(ctime),max(ctime) from {}_historyouterstatustable'.format(DeviceFlag)
    resDD = pd.read_sql_query(sqlStr, con=engine)
    # print('BeginTime{} \nEndTime{}'.format(BeginTime, EndTime))
    if BeginTime is None:
        BeginTime = resDD.ix[0, 0]  # 数据库中的最小插入shiujian
    if EndTime is None:
        EndTime = resDD.ix[0, 1]
    Begin = pd.Timestamp(BeginTime)
    End = pd.Timestamp(EndTime)
    # print('The DB Min Time is: {} \nThe DB Max Time is: {}'.format(resDD.ix[0, 0], resDD.ix[0, 1]))
    if Begin > End:
        EndTime , BeginTime= BeginTime, EndTime
        Begin, End = End, Begin

    if Begin < resDD.ix[0, 0] or Begin > resDD.ix[0, 1]:
        BeginTime = resDD.ix[0, 0]
    if EndTime is None or End > resDD.ix[0, 1] or End < resDD.ix[0, 0]:
        EndTime = resDD.ix[0, 1]
    # print('BeginTime{} \nEndTime{}'.format(BeginTime, EndTime))
    if len(MacList) == 0:
        allMacFlag = 1  # 值为1 时，获取所有的数据。
    if allMacFlag == 0:
        for table in tableNameDic.keys():
            sqlStr = "select * from {}_history{}table where ctime between \"{}\" and \"{}\" and mac in (" \
                .format(DeviceFlag, table, BeginTime, EndTime)
            for mac in MacList:
                sqlStr += "\"{}\",".format(mac)
            listStr = list(sqlStr)
            listStr.pop()
            sqlStr = "".join(listStr)
            sqlStr += ')'
            sqlStrDic[table] = sqlStr
    else:
        print(BeginTime, EndTime)
        for table in tableNameDic.keys():
            sqlStrDic[table] = "select * from {}_history{}table where ctime between \"{}\" and \"{}\""\
                .format(DeviceFlag, table, BeginTime, EndTime)
    resDic = {}
    for sqlStr in sqlStrDic.keys():
        resDic[sqlStr] = pd.read_sql_query(sqlStrDic[sqlStr], con=engine)

    del engine
    return resDic

def get_local_data_csv(SavePath='./', MacList=[], BeginTime=None, EndTime=None, DeviceFlag='aircondition'):
    res = get_local_data(MacList, BeginTime, EndTime, DeviceFlag)
    for key, data in res.items():
        data.to_csv(SavePath + key + '.csv', encoding='utf-8', mode='w')

def get_real_data(raw_data):
    def normalize(data):
        Max_temp = []
        Min_temp = []
        Max_data = []
        Min_data = []
        for i in range(len(data[0][1][0])):
            Max_temp.append([])
            Min_temp.append([])
            for j in range(len(data)):
                Max_temp[-1].append((np.array(data[j][1])[:,i]).max())
                Min_temp[-1].append((np.array(data[j][1])[:,i]).min())
        for i in range(len(Max_temp)):
            Max_data.append(np.array(Max_temp[i]).max())
            Min_data.append(np.array(Min_temp[i]).min())
        for i in range(len(data)):
            for j in range(len(data[i][1])):
                for k in range(len(data[0][1][0])):
                    data[i][1][j][k]=(data[i][1][j][k]-Min_data[k])/(Max_data[k]-Min_data[k])

    def increase_dimensional(data,min_length,interval):
        Data=[]
        Device=[]
        Virtual=[]
        for i in range(len(data)):
            _len=len(data[i][1])
            while _len >= min_length:
                temp = 6
                Virtual.append(temp)
                temp_data = []
                ID_data = []
                temp_data.append(data[i][1][(_len-min_length):_len:interval])
                ID_data = [item for sublist in temp_data for item in sublist]
                ID_data = [item for sublist in ID_data for item in sublist]
                Data.append(ID_data)
                b=data[i][0][0].split('_')[1]
                c=b.split('-')[0]
                Device.append(c)
                _len-=1
        return Data, Virtual, Device

    normalize(raw_data)
    data, virtual, device = increase_dimensional(raw_data,30,5)
    return data, virtual, device

def get_data(data,raw_data,name_str):
    ##添加除湿机数据模块##
    flag=0
    start=0
    flag_count=0
    flag1=0
    num=1
    timer=1
    for i in range(len(data['innerstatus'].snhjwd)):
        if(data['outerstatus'].ysjkgzt[i]=="1" and flag1==0):
            if((i+24)>len(data['innerstatus'].snhjwd)):
                break
            for j in range(30):
                if(data['outerstatus'].ysjkgzt[i+j]=="0"):
                    flag_count=1
                    break
            if flag_count==1:
                flag_count=0
                continue
        if(data['outerstatus'].ysjkgzt[i]=="0") or data['outerstatus'].ysjkgzt[i]==None:
            flag=1
            flag1=0
            continue

        if(data['outerstatus'].ysjkgzt[i]=="1" and (flag==1)or start==0):
            name_str1=name_str+"--"+str(num)
            raw_data.append([])
            raw_data[-1].append([])
            raw_data[-1][-1].append(name_str1)
            raw_data[-1].append([])
            flag=0
            start=1
            flag1=1
            num=int(num)+1
            timer=1
        if(data['outerstatus'].swpqwd[i]==None)or(data['innerstatus'].snhjwd[i]==None)or(data['innerstatus'].snzfqzjwd[i]==None)or(data['innerstatus'].snhjsd[i]==None):
            print("some number is None")
            continue
        if(data['outerstatus'].swpqwd[i]=="-40")or(data['innerstatus'].snhjwd[i]=="-40")or(data['innerstatus'].snzfqzjwd[i]=="-40")or(data['innerstatus'].snhjsd[i]=="0"):
            print("some number is 默认值")
            continue
        raw_data[-1][-1].append([(float)(data['innerstatus'].snhjwd[i]),(float)(data['innerstatus'].snhjsd[i]),(float)(data['outerstatus'].swpqwd[i]),(float)(data['innerstatus'].snzfqzjwd[i]),math.log(timer)])
        timer=timer+10
    return raw_data
