# %%
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import numpy as np
import sqlite3
from typing import Tuple
import sqlite3
import os
import pickle
from config import *

# %%
db_name = 'database/item_tracker.db' #物品参数，可查看Item类
db2_name = 'database/event.db'  #日志，包含物品新增、删除、移动、恢复等
model_name = 'model/yolov8n.pt' #采用的yolo模型
debug_mode = False
show_img = False
model = YOLO(model_name)
names = model.names

# %%
def input_choose_to_int_num(info:str,choose:list) ->int:
    """将输入选择转化为int

    Args:
        info (str) : 介绍信息
        choose (list): 输入信息。[str,str,str...]

    Returns:
        int: int
    """
    print('-----------------------')
    print(info)
    while(True):
        choose_str = ''
        for i in range(choose.__len__()):
            choose_str += f'{i} - '+choose[i]+'  |'
        print(choose_str)
        string = input()
        print('-----------------------')
        if(string.isdigit()):
            num = int(string)
            if (num>=0 and num<choose.__len__()):
                return num
            else:
                print('你输入的数字不在范围内！请重新输入！')
        else:
            print('你输入的不是数字！请重新输入！')

def input_to_int_num(info:str) ->int:
    """将输入转化为单个int

    Args:
        info (str): 描述文件

    Returns:
        int: int
    """
    print('-----------------------')
    print(info)
    while(True):
        string = input()
        print('-----------------------')
        if(string.isdigit()):
            num = int(string)
            return num
        else:
            print('你输入的不是数字！请重新输入！')

def input_cls_or_name_to_int_num(info:str) ->int:
    """将输入的cls 数字或name转化为 cls int

    Args:
        info (str): 说明信息

    Returns:
        int: int
    """
    print('-----------------------')
    print(info)
    print(f'输入范围为0-{len(names)-1}')
    while(True):
        string = input()
        print('-----------------------')
        if(string.isdigit()): #如果输入的是数字
            num = int(string)
            if(num>=0 and num<len(names)):
                return num
            else:
                print(f'你输入的数字{num}不在范围0-{len(names)-1}之间，请重新输入！')

        else: #如果输入的是不是数字
            for i in range(len(names)):
                if(names[i] == string):
                    return int(i)

            print('你输入的种类名称没有找到！')

def input_str_to_time() ->float:
    """将输入的str转化为时间戳

    Returns:
        float: 时间戳
    """
    while(True):
        print('-----------------------')
        print('请输入 2023-01-01 12:00:00 格式的时间')
        time_string = input()
        print('-----------------------')
        try:
            return(time.mktime(time.strptime(time_string,"%Y-%m-%d %H:%M:%S")))
        except:
            print(f'你输入的 {time_string} 格式不对！')
        
def input_str_to_tag(info:str) ->str:
    """将输入的str转化为tag，若为空白则重新输入

    Args:
        info (str): 说明信息
        
    Returns:
        str: tag
    """
    print('-----------------------')
    print(info)
    while(True):
        tag_string = input()
        print('-----------------------')
        if(tag_string.split().__len__()>0):
            return tag_string
        print('请不要输入空白字符！请重新输入！')
    
def db_execute(db_name:str,sql_string:str):
    """执行sql语句

    Args:
        db_name (str): 数据库路径
        sql_string (str): sql语句
    """
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    i = 5
    sql_success=False
    while(i>0):
        try:
            res = cur.execute(sql_string)
            result = res.fetchall()
            cur.close()
            con.commit()
            con.close()
            i = 0
            sql_success = True
            return result
        
        except:
            i -=1
    if(sql_success==False):
        cur.close()
        con.commit()
        con.close()
        raise Exception(f'数据库连接失败！')
    
def show_event(event_list:list) ->None:
    for event in event_list:
        time_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(event[10]))
        if(event[1] == 'new' ):
            print(f'hid = {event[0]:{8}} | New    | New    {names[event[3]]} uid={event[2]} at x={event[4]:.{2}} y={event[5]:.{2}} time = {time_str}')
        elif(event[1] == 'remove' ):
            print(f'hid = {event[0]:{8}} | Remove | Remove {names[event[3]]} uid={event[2]} at x={event[4]:.{2}} y={event[5]:.{2}} time = {time_str}')
        elif(event[1] == 'appear' ):
            print(f'hid = {event[0]:{8}} | Appear | Item   {names[event[3]]} uid={event[2]} appear at x={event[4]:.{2}} y={event[5]:.{2}} time = {time_str}')
            if(event[8] > 0):
                print(f'It might because of Item uid = {event[8]}')
        elif(event[1] == 'cover' ):
            print(f'hid = {event[0]:{8}} | Cover  | Item   {names[event[3]]} uid={event[2]} disappear at x={event[4]:.{2}} y={event[5]:.{2}} time = {time_str}')
            if(event[8] > 0):
                print(f'It might because of Item uid = {event[8]}')
        elif(event[1] == 'move' ):
            print(f'hid = {event[0]:{8}} | Move   | Item   {names[event[3]]} uid={event[2]} is move from x={event[4]:.{2}} y={event[5]:.{2}} to x={event[6]:.{2}} y={event[7]:.{2}} time = {time_str}')
            if(event[8] > 0):
                print(f'It might because of Item uid = {event[8]}')

# %%
info_start = '请选择要进行的操作'
choose_list_start = ['搜索物品','增加或编辑tag','删除tag','查询记录','退出程序']
info_range = '请选择搜索范围'
choose_list_range = ['搜索全部','只搜索仍在屏幕上的物体','只搜索不在屏幕上的物体']
info_how_to_search = '请输入查询依据'
choose_list_how_to_search = ['uid','cls','time','tag']
info_if_add_search_condition = '是否增加搜索条件'
choose_list_if_add_search_condition = ['否','是']
info_if_left_interval = '是否添加时间左区间'
choose_list_if_left_interval = ['不添加(查询范围从数据库建立开始)','添加(查询范围从此时间开始)']
info_if_right_interval = '是否添加时间右区间'
choose_list_if_right_interval = ['不添加(查询范围一直到现在)','添加(查询范围从此时间结束)']


# %%
if __name__=='__main__':
    while(True):
        start_num = input_choose_to_int_num(info_start,choose_list_start)
        if(start_num==0): #选择搜索物品
            sql_search_str ='SELECT * FROM item '
            range_num = input_choose_to_int_num(info_range,choose_list_range)
            if(range_num == 1): #只搜索仍在屏幕上的物体
                sql_search_str +='WHERE status = 1 and '
            elif(range_num == 2): #只搜索不在屏幕上的物体
                sql_search_str +='WHERE status = 0 and '
            elif(range_num == 0): #搜索全部
                sql_search_str +=''
            else:
                print(f'start_num = {start_num},break!')
                break
            
            while(input_choose_to_int_num(info_if_add_search_condition,choose_list_if_add_search_condition) == 1):#是否增加搜索条件
                how_to_search_num = input_choose_to_int_num(info_how_to_search,choose_list_how_to_search)
                if(how_to_search_num == 0): #uid
                    uid_num = input_to_int_num('请输入要查询的uid')
                    sql_search_str += f'WHERE uid = {uid_num} and '
                elif(how_to_search_num == 1): #cls
                    cls_num = input_cls_or_name_to_int_num('请输入要查询的cls数字或是名称')
                    sql_search_str += f'WHERE cls = {cls_num} and '
                elif(how_to_search_num == 2): #time
                    left_num = input_choose_to_int_num(info_if_left_interval,choose_list_if_left_interval)
                    if(left_num == 1):
                        left_time = input_str_to_time()
                        sql_search_str += f'WHERE utc >= {left_time} and '

                    right_num = input_choose_to_int_num(info_if_right_interval,choose_list_if_right_interval)
                    if(right_num == 1):
                        right_time = input_str_to_time()
                        sql_search_str += f'WHERE utc <= {right_time} and '

                elif(how_to_search_num == 3): #tag
                    
                    tag_str = input_str_to_tag('请输入要搜索的tag')
                    sql_search_str += f'WHERE tag like "%{tag_str}%" and '

                else:
                    raise Exception(f'how_to_search_num = {how_to_search_num},超出范围！')
            if(sql_search_str[-4:-1]=='and'): #删除多余的and
                sql_search_str = sql_search_str[:-4]

            res = db_execute(db_name,sql_search_str)
            item_list = []
            if(debug_mode):
                print(sql_search_str)
            print('uid\tcls\t\tx    y    w    h\tconf\ttag\tstatus')
            for k,i in enumerate(res):
                j = Item(*i)
                print(f'{j.uid}\t{names[j.cls]:{15}}{j.x:.{2}} {j.y:.{2}}  {j.w:.{2}} {j.h:.{2}}\t{j.conf:.{2}}\t{j.tag}\t{j.status}')
                if(show_img):
                    plt.figure(dpi=300)
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow((pickle.loads(j.fig)))
                    plt.show()
            print('***********************')
        elif(start_num==1): #增加tag
            uid_num = input_to_int_num('请输入要增加或编辑tag的uid')
            tag_str = input_str_to_tag('请输入要增加或编辑的tag内容')
            sql_update_tag = f"UPDATE item SET tag='{tag_str}' WHERE uid = {uid_num}"
            res = db_execute(db_name,sql_update_tag)
            if(debug_mode):
                print(sql_update_tag)
            print('**********************************************')
        elif(start_num==2): #删除tag
            uid_num = input_to_int_num('请输入要删除tag的uid')
            sql_update_tag = f"UPDATE item SET tag=NULL WHERE uid = {uid_num}"
            res = db_execute(db_name,sql_update_tag)
            if(debug_mode):
                print(sql_update_tag)
            print('**********************************************')
        elif(start_num==3): #查询记录
            uid_num = input_to_int_num('请输入要查询记录的uid')
            left_num = input_choose_to_int_num(info_if_left_interval,choose_list_if_left_interval)
            sql_event = f'SELECT * FROM event WHERE uid ={uid_num} and '
            if(left_num == 1):
                left_time = input_str_to_time()
                sql_event += f'WHERE utc >= {left_time} and '

            right_num = input_choose_to_int_num(info_if_right_interval,choose_list_if_right_interval)
            if(right_num == 1):
                right_time = input_str_to_time()
                sql_event += f'WHERE utc <= {right_time} and '
            if(sql_event[-4:-1]=='and'): #删除多余的and
                sql_event = sql_event[:-4]

            res = db_execute(db2_name,sql_event)
            if(debug_mode):
                print(sql_event)
            show_event(res)
            print('**********************************************')
        elif(start_num==4): #退出
            break
        else:
            raise Exception(f'start_num = {start_num},超出范围！')

        

# %%


# %%



