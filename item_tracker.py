# %%
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import sqlite3
from typing import Tuple
import sqlite3
import os
import pickle
from PIL import ImageDraw,Image,ImageFont
from multiprocessing import Process,Queue

# %%
if not os.path.exists('database'):
    os.makedirs('database')

# %%
# 程序参数  距离为归一化距离
db_name = 'database/item_tracker.db' #物品参数，可查看Item类
db2_name = 'database/event.db'  #日志，包含物品新增、删除、移动、恢复等
model_name = 'model/yolov8s.pt' #采用的yolo模型
confidence = 0.6 #置信度，高于此值才认为识别到了该物体，值越高识别到的物体越少
edge_range = [0.1,0.2,0.025] #物体中心离边界距离小于[0]一定算在边界,大于[1]一定不在边界,其余情况下物体边缘与边界之间的缝隙小于[2]算在边界
recover_distance = 0.03 #新识别到的物体若与已被删除的旧物体小于此距离可认为是同一物体
close_correlate = 1.1 #两个物体之间的相关系数，用于判断一个物体新增、消失是否受其他物体的影响，越大认为二者是相关的
stable_distance = 0.05 #一个物体在多少范围内认为是不动的

waiting_num_add = 3 #一个新出现的物体连续多少帧出现才加入数据库
waiting_time_add = 0.5 #一个新出现的物体出现多久才加入数据库，单位秒，两个条件要同时满足

add_range = [0,0.15] #物品可能是移动着进入的,在此范围内认为物品是新增
overlapping_rate = 0.4 #覆盖率低于overlapping_rate的item

# %%
class Item:
    def __init__(self,uid:int,cls:int,x:float,y:float,w:float,h:float,conf:float,fig:np.array,utc:float,tag:str = None,status:int=1):
        """Item类

        Args:
            uid (int): 物品序号，唯一
            cls (int): 物品类别
            x (float): 物品中心x坐标
            y (float): 物品中心y坐标
            w (float): 物品宽度
            h (float): 物品高度
            conf (float): 置信度
            fig (np.array): 物品照片
            utc (float): 物品最后修改时间\物品被识别时间\物品移动时间
            tag (str, optional): 物品tag
            status (int, optional): 物品状态，1为存在，0为移除 . Defaults to 1.
        """
        self.uid = uid # 物品序号，唯一
        self.cls = cls # 物品类别
        self.x = x # 物品中心x坐标
        self.y = y # 物品中心y坐标
        self.w = w # 物品宽度
        self.h = h # 物品高度
        self.conf = conf # 置信度
        self.fig = fig # 物品照片
        self.utc = utc # 物品最后修改时间\物品被识别时间\物品移动时间
        self.tag = tag # 物品tag
        self.status = status # 物品状态，1为存在，0为移除 

class Uid:
    """快捷生成id
    """
    uid = 0
    def __init__(self,initial_uid:int = 0) -> None:
        self.uid = initial_uid

    def getuid(self) -> int:
        self.uid += 1
        return(self.uid)
    
class TimeRunning:
    def __init__(self,tag:str) -> None:
        """用以获取各模块消耗时间

        Args:
            tag (str): 模块名称
        """
        self.tag = tag
        self.last_time = time.time()
    def get_init_time(self):
        self.last_time = time.time()
    def get_delta_time(self):
        """
        Purpose: one
        """
        now_time = time.time()
        delta_time = now_time - self.last_time
        return(delta_time)
        
class Database:
    """数据库类，封装好了sqlite函数

    Returns:
        _type_: _description_
    """

    console_output = False

    def __init__(self,names:list,db_name:str = 'item_tracker.db',db2_name:str = 'event.db',console_output:bool = False) -> None:
        """初始化数据库

        Args:
            db_name (str, optional): item列表存储数据库. Defaults to 'item_tracker.db'.
            db2_name (str, optional): event列表存储数据库. Defaults to 'event.db'.
            console_output (bool, optional): 控制台是否显示event. Defaults to False.
        """
        self.names = names  
        self.db_name = db_name
        self.db2_name = db2_name
        self.console_output = console_output
        if(os.path.exists(self.db_name) == False): #给定位置没找到数据库就新建一个
            con = sqlite3.connect(self.db_name)
            cur = con.cursor()
            cur.execute('''CREATE TABLE item 
                        (
                        uid INTEGER PRIMARY KEY AUTOINCREMENT,
                        cls INTEGER,
                        x FLOAT,
                        y FLOAT,
                        w FLOAT,
                        h FLOAT,
                        conf FLOAT,
                        fig BLOB,
                        utc FLOAT,
                        tag TEXT,
                        status INTEGER
                        )
                        ''')
            cur.close()
            con.commit()
            con.close()

            con = sqlite3.connect(self.db2_name)
            cur = con.cursor()
            cur.execute('''CREATE TABLE event 
                        (
                        hid INTEGER PRIMARY KEY AUTOINCREMENT,
                        type TEXT,
                        uid INTEGER,
                        cls INTEGER,
                        arg1 FLOAT,
                        arg2 FLOAT,
                        arg3 FLOAT,
                        arg4 FLOAT,
                        arg5 INT,
                        fig BLOB,
                        utc FLOAT
                        )
                        ''')
            cur.close()
            con.commit()
            con.close()
        
    def db_write_event(self,event:list) -> None:
        """写日志
        
        Args:
            event (list): [type:str,uid:int,cls:int,arg1-arg5,fig:np.array,utc:float]
                    type    |                                   parameter
                new         | uid   cls     x_add     y_add     None        None        None            fig     utc
                remove      | uid   cls     x_remove  y_remove  None        None        None            fig     utc
                appear      | uid   cls     x_appear  y_appear  None        None        item_uid_round  fig     utc
                cover       | uid   cls     x_cover   y_cover   None        None        item_uid_round  fig     utc
                move        | uid   cls     x_s       y_s       x_e         y_e         item_uid_round  fig     utc
                error       | None  None    None      None      None        None        None            event   utc
        """
        console_output = self.console_output
        con = sqlite3.connect(self.db2_name)
        cur = con.cursor()
        if (event[0] == 'new' ):
            param = (str(event[0]),int(event[1]),int(event[2]),float(event[3]),float(event[4]),pickle.dumps(event[5]),time.time())
            cur.execute('''INSERT INTO event(type,uid,cls,arg1,arg2,fig,utc) VALUES(?,?,?,?,?,?,?)''',param)
            if (console_output):
                print('New    | New item ',self.names[param[2]],' uid = ',event[1],' at ',event[3],event[4])

        elif (event[0] == 'remove' ):
            param = (str(event[0]),int(event[1]),int(event[2]),float(event[3]),float(event[4]),pickle.dumps(event[5]),time.time())
            cur.execute('''INSERT INTO event(type,uid,cls,arg1,arg2,fig,utc) VALUES(?,?,?,?,?,?,?)''',param)
            if (console_output):
                print('Remove | Remove item ',self.names[param[2]],' uid = ',event[1],'from ',event[3],event[4])

        elif (event[0] == 'appear' ):
            param = (str(event[0]),int(event[1]),int(event[2]),float(event[3]),float(event[4]),int(event[5]),pickle.dumps(event[6]),time.time())
            cur.execute('''INSERT INTO event(type,uid,cls,arg1,arg2,arg5,fig,utc) VALUES(?,?,?,?,?,?,?,?)''',param)
            if (console_output):
                if event[5]<0:
                    sentence = 'it is very strange'
                else:
                    sentence = 'it might be take out from item uid = '+str(event[5])
                print('Appear | Item ',self.names[param[2]],' uid = ',event[1],' appear at',event[3],event[4],' , ',sentence)
                
        elif (event[0] == 'cover' ):
            param = (str(event[0]),int(event[1]),int(event[2]),float(event[3]),float(event[4]),int(event[5]),pickle.dumps(event[6]),time.time())
            cur.execute('''INSERT INTO event(type,uid,cls,arg1,arg2,arg5,fig,utc) VALUES(?,?,?,?,?,?,?,?)''',param)
            if (console_output):
                if event[5]<0:
                    sentence = 'it is very strange'
                else:
                    sentence = 'it might be cover by item uid = '+str(event[5])
                print('Cover  | Item ',self.names[param[2]],' uid = ',event[1],' disappear at ',event[3],event[4],' , ',sentence)
        
        elif (event[0] == 'move' ):
            param = (str(event[0]),int(event[1]),int(event[2]),float(event[3]),float(event[4]),float(event[5]),float(event[6]),int(event[7]),pickle.dumps(event[8]),time.time())
            cur.execute('''INSERT INTO event(type,uid,cls,arg1,arg2,arg3,arg4,arg5,fig,utc) VALUES(?,?,?,?,?,?,?,?,?,?)''',param)
            if (console_output):
                if event[7]<0:
                    sentence = 'it can move by itself'
                else:
                    sentence = 'it was pushed by item uid = '+str(event[7])
                print('Move   | Item ',self.names[param[2]],' uid = ',event[1],' is move from',event[3],event[4],' to ',event[5],event[6],',might because ',sentence)
        
        else:
            param = ['error',pickle.dumps(event),time.time()]
            cur.execute('''INSERT INTO event(type,fig,utc) VALUES(?,?,?)''',param)
            if (console_output):
                print('Error  | Error event write in!')

            
        
        cur.close()
        con.commit()
        con.close()

    def db_save(self,item_add_list) -> None:
        """加入无相关新物体

        Args:
            item_add_list (_type_): item_add_list
        """
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()
        for i in item_add_list:
            param = (int(i.cls),float(i.x),float(i.y),float(i.w),float(i.h),float(i.conf),pickle.dumps(i.fig),time.time(),None,int(1))
            cur.execute('''INSERT INTO item(cls,x,y,w,h,conf,fig,utc,tag,status) VALUES(?,?,?,?,?,?,?,?,?,?)''',param)
            event = ['new',int(cur.lastrowid),param[0],param[1],param[2],i.fig]
            self.db_write_event(event)
        cur.close()
        con.commit()
        con.close()
    
    def db_appear(self,item_add_list:list,item_correlate_uid_list: list) -> None:
        """加入有相关新物体，可能是突然出现，也可能是从其他物体中拿出、其他物体移除遮挡

        Args:
            item_add_list (list): item_add_list
            item_correlate_uid_list (list): 相关物体uidlist
        """
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()
        for k in range(len(item_add_list)):
            i = item_add_list[k]
            j = item_correlate_uid_list[k]

            param = (int(i.cls),float(i.x),float(i.y),float(i.w),float(i.h),float(i.conf),pickle.dumps(i.fig),time.time(),None,int(1))
            cur.execute('''INSERT INTO item(cls,x,y,w,h,conf,fig,utc,tag,status) VALUES(?,?,?,?,?,?,?,?,?,?)''',param)
            event = ['appear',int(cur.lastrowid),param[0],param[1],param[2],j,i.fig]
            self.db_write_event(event)
        cur.close()
        con.commit()
        con.close()

    def db_read_exist(self) -> list:
        """读取物品表中存活的数据

        Returns:
            list: 物品表中存活的数据
        """
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()
        res = cur.execute('''SELECT * FROM item WHERE status = 1''')
        temp = res.fetchall()
        item_db_0 = []
        for i in temp:
            j = list(i)
            j[7] = pickle.loads(j[7])
            item_db_0.append(Item(*j))
        cur.close()
        con.commit()
        con.close()
        return item_db_0
    
    def db_read_remove(self) -> list: 
        """读取物品表中被移除的数据

        Returns:
            list: 物品表中被移除的数据
        """
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()
        res = cur.execute('''SELECT * FROM item WHERE status = 0''')
        temp = res.fetchall()
        item_db_remove_0 = []
        for i in temp:
            j = list(i)
            j[7] = pickle.loads(j[7])
            item_db_remove_0.append(Item(*j))
        cur.close()
        con.commit()
        con.close()
        return item_db_remove_0
    
    def db_update(self,item_old_list:list,item_new_list:list,item_correlate_uid_list: list) -> None:
        """更新物品(物品移动)

        Args:
            item_old_list (list): 旧物品序列
            item_new_list (list): 新物品序列
            item_correlate_uid_list (list): 相关物品uid list
        """
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()
        for j in range(item_old_list.__len__()):
            i = item_new_list[j]
            k = item_old_list[j]
            l = item_correlate_uid_list[j]
            param = (int(i.cls),float(i.x),float(i.y),float(i.w),float(i.h),float(i.conf),pickle.dumps(i.fig),time.time(),int(item_old_list[j].uid))
            cur.execute('''
                              UPDATE item SET 
                              cls = ? ,
                              x = ? ,
                              y = ? ,
                              w = ? ,
                              h = ? ,
                              conf = ? ,
                              fig = ? ,
                              utc = ? WHERE uid = ? ''',param)
            event = ['move',k.uid,i.cls,k.x,k.y,i.x,i.y,l,i.fig]
            self.db_write_event(event)

        cur.close()
        con.commit()
        con.close()

    def db_remove(self,item_remove_list:list) -> None:
        """将list的Item status 置为1

        Args:
            item_remove_list (list[Item,Item,Item...]): item_remove_list
        """


        con = sqlite3.connect(self.db_name)
        cur = con.cursor()
        for i in item_remove_list:
            res = cur.execute('''
                    SELECT status FROM item WHERE (uid = ?)
                    ''',[i.uid])
            status = res.fetchone()[0]
            if (status):
                cur.execute('''
                                UPDATE item SET status = 0 WHERE (uid = ?)
                                ''',[i.uid])
                event = ['remove',i.uid,i.cls,i.x,i.y,i.fig]
                self.db_write_event(event)
        cur.close()
        con.commit()
        con.close()

    def db_cover(self,item_remove_list:list,item_correlate_uid_list: list) -> None:
        """将list对应的Item认为被相关物体覆盖/突然消失

        Args:
            item_remove_list (list): 删除Item list 对应的item
            item_remove_list (list): 删除Item的相关uid list
            item_correlate_uid_list (list): 相关物品uid list
        """


        con = sqlite3.connect(self.db_name)
        cur = con.cursor()
        for k in range(len(item_remove_list)):
            i = item_remove_list[k]
            res = cur.execute('''
                    SELECT status FROM item WHERE (uid = ?)
                    ''',[i.uid])
            status = res.fetchone()[0]
            if (status):
                cur.execute('''
                                UPDATE item SET status = 0 WHERE (uid = ?)
                                ''',[i.uid])
                event = ['cover',i.uid,i.cls,i.x,i.y,item_correlate_uid_list[k],i.fig]
                self.db_write_event(event)
        cur.close()
        con.commit()
        con.close()

    def db_recover(self,item_recover_uid_list:list,item_new_list:list,item_correlate_uid_list: list) -> None:
        """恢复对应uid列表的Item

        Args:
            item_recover_uid_list (list): 已被删除的item uid list
            item_new_list (list): 用以替代被删除item list
            item_correlate_uid_list (list): 相关物品uid list
        """
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()
        for k in range(len(item_recover_uid_list)):
            j = item_recover_uid_list[k] # uid
            i = item_new_list[k]
            l = item_correlate_uid_list[k]
            cur.execute('''
                              UPDATE item SET status = 1 WHERE (uid = ?)
                              ''',[j])

            param = (int(i.cls),float(i.x),float(i.y),float(i.w),float(i.h),float(i.conf),pickle.dumps(i.fig),time.time(),int(j))
            cur.execute('''
                              UPDATE item SET 
                              cls = ? ,
                              x = ? ,
                              y = ? ,
                              w = ? ,
                              h = ? ,
                              conf = ? ,
                              fig = ? ,
                              utc = ? WHERE uid = ? ''',param)
            
            event = ['appear',j,i.cls,i.x,i.y,l,param[6]]
            self.db_write_event(event)


        cur.close()
        con.commit()
        con.close()


# %%
def frame2item_array(frame:np.ndarray,model:YOLO,confidence:float=confidence) -> list:
    """将摄像头获取到的图像转化为物品信息

    Args:
        frame (np.ndarray): 摄像头输入的视频帧
        model (YOLO): 采用的YOLO模型
        confidence (float, optional): 识别置信度 低于此值的将不被认为是物品   0-1 Defaults to 0.5.

    Returns:
        list
    """

    img = frame[...,::-1]
    results = model.predict(img,conf = confidence,verbose = False)
    result = results[0]
    boxes = result.boxes
    cls = boxes.cls.numpy()
    cls = np.array(cls,dtype=int)
    xywhn = boxes.xywhn.numpy()
    xywhn = np.array(xywhn)
    xyxy = boxes.xyxy.numpy()
    xyxy = np.array(xyxy,dtype=int)
    conf = boxes.conf.numpy()
    conf_index = np.where(conf>=confidence,True,False)
    cls = cls[conf_index]
    xywhn = xywhn[conf_index]
    xyxy = xyxy[conf_index]
    conf = conf[conf_index]

        

    item_frame_0 = []
    temp_uid = -1
    for i in range(len(cls)):
            box_img = img[xyxy[i][1]:xyxy[i][3],xyxy[i][0]:xyxy[i][2],:]
            item_frame_0.append(Item(temp_uid,cls[i],xywhn[i][0],xywhn[i][1],xywhn[i][2],xywhn[i][3],conf[i],box_img,time.time()))
    return item_frame_0

def draw_img(frame: np.ndarray,item_db_0: list,names:list,time_list:float) -> np.ndarray:
    """返回渲染好的图片

    Args:
        frame (np.ndarray): 原始图像
        item_db_0 (list): db
        names (list): names
        time_list (float): 各组件用时

    Returns:
        np.ndarray: 图像
    """
    fig = frame
    img = Image.fromarray(fig)
    draw = ImageDraw.Draw(img)
    color = [(255,0,0),(255,255,0),(0,0,255),(0,255,255),(0,255,0),(255,0,255)]
    color_num = 0
    for i in item_db_0:
        color_num = i.uid%len(color)
        x1 = (i.x-i.w/2)*fig.shape[1]
        y1 = (i.y-i.h/2)*fig.shape[0]
        x2 = (i.x+i.w/2)*fig.shape[1]
        y2 = (i.y+i.h/2)*fig.shape[0]
        draw.rectangle([x1,y1,x2,y2],outline = color[color_num])
        if(i.tag == None):
            text = str(str(i.uid)+"--"+names[i.cls])+"---"+str(int(i.conf*100))
        else:
            text = str(str(i.uid)+"--"+names[i.cls])+"---"+str(int(i.conf*100))+"   "+str(i.tag)
        font_size = int(max(min(70*i.w,40),20))
        font_dynamic = ImageFont.truetype('font.ttf',size = font_size)
        draw.text([x1,y1],text,fill = color[color_num],font=font_dynamic)
        color_num += 1
        
    font_hint = ImageFont.truetype('font.ttf',size = 30)
    font_fps = ImageFont.truetype('font.ttf',size = 30)
    text_hint = ["cam","YOLO","data","all","fps"]
    text_time_list = [str(int(1000*time_list[1])),str(int(1000*time_list[2])),str(int(1000*time_list[3])),str(int(1000*time_list[0])),str(int(10*max(1/time_list[0],1))/10)]
    for i in range(len(text_hint)):
        draw.text([fig.shape[1]*(0.55+i*0.1),fig.shape[0]*0.87],str(text_hint[i]),fill = (0,255,0),font=font_hint)
        draw.text([fig.shape[1]*(0.55+i*0.1),fig.shape[0]*0.94],str(text_time_list[i]),fill = (0,255,0),font=font_fps)
    picture = np.array(img)
    return picture

def item_distance(item1:Item,item2:Item) ->float:
    """计算两个item之间的距离

    Args:
        item1 (Item): Item实例
        item2 (Item): Item实例

    Returns:
        float: 距离值
    """
    return(((item1.x-item2.x)**2+(item1.y-item2.y)**2)**0.5)

def item_if_in_distance(item1:Item,item2:Item,distance:float=0.01) ->bool:
    """判断两个物体是否小于给定距离

    Args:
        item1 (Item): item1
        item2 (Item): item2
        sigma (float, optional): 判断值. Defaults to 0.01.

    Returns:
        bool: _description_
    """
    return(item_distance(item1,item2)<=distance)

def item_if_near_edge(item:Item,edge:list = edge_range) ->bool:
    """判断物体是否靠近边缘

    Args:
        item (Item): item
        edge (list, optional): 物体中心小于[0]一定算在边界,大于[1]一定不在边界,其余情况缝隙小于[2]算在边界. Defaults to [0.1,0.2,0.025].

    Returns:
        bool: item_if_near_edge
    """
    item_edge_distance = min(min((1-item.x),item.x),min((1-item.y),item.y))
    crack_distance = min(min((1-item.x-item.w/2),(item.x-item.w/2)),min((1-item.y-item.h/2),(item.y+item.h/2)))
    if(item_edge_distance<=edge[0]):
        return(True)
    elif(item_edge_distance>=edge[1]):
        return(False)
    else:
        return(crack_distance<=edge[2])

def item_overlapping_rate(item1:Item,item2:Item) ->float:
    """两个物体之间的覆盖率,定义为重叠面积除以二者面积中的小值

    Args:
        item1 (Item): Item
        item2 (Item): Item

    Returns:
        float: 0-1
    """
    if((np.abs(item1.x-item2.x)>=(item1.w+item2.w)/2)or(np.abs(item1.y-item2.y)>=(item1.h+item2.h)/2)):
        return 0.00
    else:
        area_overlapping = min(min(item1.w,item2.w),((item1.w+item2.w)/2-np.abs(item1.x-item2.x)))*min(min(item1.h,item2.h),((item1.h+item2.h)/2-np.abs(item1.y-item2.y)))
        return area_overlapping/min(item1.w*item1.h,item2.w*item2.h)

def item_if_in_range(item1:Item,item2:Item,range:list) ->bool:
    """两个物体距离是否在range范围内

    Args:
        item1 (Item): item1
        item2 (Item): item2
        range (list): [a,b]，距离需在a、b之间

    Returns:
        bool: 
    """
    return(item_distance(item1,item2)<=range[1] and item_distance(item1,item2) >= range[0])

def item_if_is_recover(item1:Item,item2:Item,distance:float=recover_distance) ->bool:#可以用prob和w h 来进一步限制
    """判断尝试recover的新旧物体是否可认为是同一物体

    Args:
        item1 (Item): item
        item2 (Item): item
        distance (float, optional): 判断距离. Defaults to 0.01.

    Returns:
        bool: _description_
    """
    # scale = (item1.w+item1.h+item2.w+item2.h)/4 #特征长度
    # real_distance = item_distance(item1,item2)
    # determin_length = real_distance*length
    return(item_distance(item1,item2)<=distance)
    
def item_if_close_correlate(item1:Item,item2:Item,item_distance:float,correlation:float=close_correlate) ->bool:#可以用prob和w h 来进一步限制
    """判断两个item之间距离是否近到是相关的

    Args:
        item1 (Item): item
        item2 (Item): item
        item_distance (float): 二者距离
        correlation (float, optional): 判断参数，越大认为二者是相关的，即return true Defaults to 1.1.

    Returns:
        bool: _description_
    """
    len1 = (item1.w**2+item1.h**2)**0.5
    len2 = (item2.w**2+item2.h**2)**0.5
    len = (len1+len2)/2
    return(item_distance<len*correlation)

def search_stable_item(item_db_0:list,item_frame_0:list,stable_distance:float=stable_distance) ->Tuple[list,list,list,list]:
    """查找数据库中不动的物体
        也可认为是查找list1与list2位移小于stable_distance的物体

    Args:
        item_db_0 (list): 数据库中物体列表
        item_frame_0 (list): frame中物体列表
        stable_distance (float, optional): 物体小于多少认为没有移动 Defaults to 0.01.

    Returns:
        Tuple[list,list,list,list]: db中不动的物体，frame中不动的物体，db中剩余物体，frame中剩余物体
    """
    item_db_stable = []
    item_frame_stable = []
    item_db_unstable = list(item_db_0)
    item_frame_unstable = list(item_frame_0)
    for item_initial in item_db_0: # 筛选不动的物体
        candidate_list = [] #足够近的名单
        for item_find in item_frame_unstable:
            if(item_if_in_distance(item_initial,item_find,stable_distance)):
                candidate_list.append([item_initial,item_find,item_distance(item_initial,item_find)])


        candidate_list.sort(key = lambda x:x[2])#以距离远近重新排序
        if_find_stable = False
        for i in candidate_list:
            if(i[0].cls==i[1].cls and if_find_stable == False):
                item_db_stable.append(i[0])
                item_db_unstable.remove(i[0])
                item_frame_stable.append(i[1])
                item_frame_unstable.remove(i[1])
                if_find_stable = True

    return(item_db_stable,item_frame_stable,item_db_unstable,item_frame_unstable)

def search_same_cls_in_range(item_db_0:list,item_list_1:list,item_list_2:list,range:list=[0.05,0.2]) ->Tuple[list,list,list,list,list]:
    """查找在两item list中有着相同cls并在range范围内的物体,并找出list1中item与db_0非相同uid中最相关的uid

    Args:
        item_sb_0 (list): 数据库中物品list
        item_list_1 (list): list1
        item_list_2 (list): list2
        move_range (list, optional): 筛选条件. Defaults to [0.03,0.2].

    Returns:
        Tuple[list,list,list,list,list]: list1中移动的物体，list2中移动的物体，list1中剩余的物体，list2中剩余的物体，move相关的uid
    """
    item_list_1_in_range = []
    item_list_2_in_range = []
    item_list_1_rest = list(item_list_1)
    item_list_2_rest = list(item_list_2)
    item_list_1_db_0_correlate = []
    for i in item_list_1:
        candidate_list = []
        for j in item_list_2_rest:
            if(item_if_in_range(i,j,range)):
                candidate_list.append([i,j,item_distance(i,j)])
        candidate_list.sort(key = lambda x:x[2])#以距离远近重新排序
        if_find_move = False
        for k in candidate_list:
            if(k[0].cls==k[1].cls and if_find_move == False):
                item_list_1_in_range.append(k[0])
                item_list_1_rest.remove(k[0])
                item_list_2_in_range.append(k[1])
                item_list_2_rest.remove(k[1])
                if_find_move = True

    for i in item_list_1_in_range:
        temp_list = []
        for j in item_db_0:#对db中非自身物体进行判断是否影响
            dist = item_distance(i,j)
            if(item_if_close_correlate(i,j,dist)):
                temp_list.append([j.uid,dist])
        temp_list.sort(key = lambda x:x[1])
            
        if(len(temp_list)!=0):
            if(i.uid==temp_list[0][0]):
                if(len(temp_list)>1):
                    item_list_1_db_0_correlate.append(temp_list[1][0])
                else:
                    item_list_1_db_0_correlate.append(-1)
            else:
                    item_list_1_db_0_correlate.append(temp_list[0][0])
        else:
                    item_list_1_db_0_correlate.append(-1)



    return(item_list_1_in_range,item_list_2_in_range,item_list_1_rest,item_list_2_rest,item_list_1_db_0_correlate)

def search_in_range(item_db_0:list,item_list_1:list,item_list_2:list,range:list=[0.05,0.2]) ->Tuple[list,list,list,list,list]:
    """查找范围内相关的物体,将db1中每个元素在db2中进行搜寻，如果够近认为是强相关元素，并找出list1中item与db_0非相同uid中最相关的uid
    Args:
        item_db_unstable (list): 数据库中非固定的list
        item_frame_unstable (list): frame中非固定的list
        move_range (list, optional): _description_. Defaults to [0.03,0.2].

    Returns:
        Tuple[list,list,list,list,list]: db中移动的物体，frame中移动的物体，db中丢失的物体，fram中新增的物体，move相关的uid
    """
    item_list_1_in_range = []
    item_list_2_in_range = []
    item_list_1_rest = list(item_list_1)
    item_list_2_rest = list(item_list_2)
    item_list_1_db_0_correlate = []
    for i in item_list_1:
        candidate_list = []
        for j in item_list_2_rest:
            if(item_if_in_range(i,j,range)):
                candidate_list.append([i,j,item_distance(i,j)])
        candidate_list.sort(key = lambda x:x[2])#以距离远近重新排序
        if_find_move = False
        for k in candidate_list:
            if(if_find_move == False):
                item_list_1_in_range.append(k[0])
                item_list_1_rest.remove(k[0])
                item_list_2_in_range.append(k[1])
                item_list_2_rest.remove(k[1])
                if_find_move = True

    for i in item_list_1_in_range:
        temp_list = []
        for j in item_db_0:#对db中非自身物体进行判断是否影响
            dist = item_distance(i,j)
            if(item_if_close_correlate(i,j,dist)):
                temp_list.append([j.uid,dist])
        temp_list.sort(key = lambda x:x[1])
            
        if(len(temp_list)!=0):
            if(i.uid==temp_list[0][0]):
                if(len(temp_list)>1):
                    item_list_1_db_0_correlate.append(temp_list[1][0])
                else:
                    item_list_1_db_0_correlate.append(-1)
            else:
                    item_list_1_db_0_correlate.append(temp_list[0][0])
        else:
                    item_list_1_db_0_correlate.append(-1)
                
                

    return(item_list_1_in_range,item_list_2_in_range,item_list_1_rest,item_list_2_rest,item_list_1_db_0_correlate)

def search_same_cls_overlapping(item_db_0:list,item_list_1:list,item_list_2:list,range:list=[0,0.2],overlapping_rate:float = overlapping_rate) ->Tuple[list,list,list,list,list]:
    """查找在两item list中有着相同cls并在range范围内的物体,且覆盖率低于overlapping_rate的item,并找出list1中item与db_0非相同uid中最相关的uid

    Args:
        item_sb_0 (list): 数据库中物品list
        item_list_1 (list): list1
        item_list_2 (list): list2
        move_range (list, optional): 筛选条件. Defaults to [0.03,0.2].

    Returns:
        Tuple[list,list,list,list,list]: list1中移动的物体，list2中移动的物体，list1中剩余的物体，list2中剩余的物体，move相关的uid
    """
    item_list_1_in_range = []
    item_list_2_in_range = []
    item_list_1_rest = list(item_list_1)
    item_list_2_rest = list(item_list_2)
    item_list_1_db_0_correlate = []
    for i in item_list_1:
        candidate_list = []
        for j in item_list_2_rest:
            if(item_if_in_range(i,j,range)):
                candidate_list.append([i,j,item_distance(i,j)])
        candidate_list.sort(key = lambda x:x[2])#以距离远近重新排序
        if_find_move = False
        for k in candidate_list:
            if(k[0].cls==k[1].cls and item_overlapping_rate(k[0],k[1])>=overlapping_rate and if_find_move == False):
                item_list_1_in_range.append(k[0])
                item_list_1_rest.remove(k[0])
                item_list_2_in_range.append(k[1])
                item_list_2_rest.remove(k[1])
                if_find_move = True

    for i in item_list_1_in_range:
        temp_list = []
        for j in item_db_0:#对db中非自身物体进行判断是否影响
            dist = item_distance(i,j)
            if(item_if_close_correlate(i,j,dist)):
                temp_list.append([j.uid,dist])
        temp_list.sort(key = lambda x:x[1])
            
        if(len(temp_list)!=0):
            if(i.uid==temp_list[0][0]):
                if(len(temp_list)>1):
                    item_list_1_db_0_correlate.append(temp_list[1][0])
                else:
                    item_list_1_db_0_correlate.append(-1)
            else:
                    item_list_1_db_0_correlate.append(temp_list[0][0])
        else:
                    item_list_1_db_0_correlate.append(-1)



    return(item_list_1_in_range,item_list_2_in_range,item_list_1_rest,item_list_2_rest,item_list_1_db_0_correlate)


def search_recover_stable_item(item_recover_list:list,remove_list:list,stable_range:float=0.05) ->Tuple[list,list,list,list]:
    """对新出现item搜索stable_range范围内是否有相同种类已经消失的物体

    Args:
        item_recover_list (list): 新出现item list.
        remove_list (list): 库中已经消失的物体list.
        stable_range (float, optional): 搜索范围. Defaults to 0.01.

    Returns:
        Tuple[list,list,list,list]: _description_
    """
    item_db_stable = []
    item_frame_stable = []
    item_db_unstable = list(item_recover_list)
    item_frame_unstable = list(remove_list)
    for item_initial in item_recover_list: # 筛选不动的物体
        candidate_list = [] #足够近的名单
        for item_find in item_frame_unstable:
            if(item_if_is_recover(item_initial,item_find,stable_range)):
                candidate_list.append([item_initial,item_find,item_distance(item_initial,item_find)])


        candidate_list.sort(key = lambda x:x[2])#以距离远近重新排序
        if_find_stable = False
        for i in candidate_list:
            if(i[0].cls==i[1].cls and if_find_stable == False):
                item_db_stable.append(i[0])
                item_db_unstable.remove(i[0])
                item_frame_stable.append(i[1])
                item_frame_unstable.remove(i[1])
                if_find_stable = True

    return(item_db_stable,item_frame_stable,item_db_unstable,item_frame_unstable)

class AddItemWaiting:
    
    def __init__(self,item:Item,start_circle_num:int,start_time:float=time.time(),waiting_num:int = waiting_num_add,waiting_time:float = waiting_time_add):
        """初始化新增物体对象

        Args:
            item (Item): 新增物体item
            start_circle_num (int): 此时循环数
            start_time (float, optional): 此时utc Defaults to time.time().
            waiting_num (int, optional): 超过多少帧后添加此物体. Defaults to 4.
            waiting_time (float, optional): 超过多少时间后添加此物体.两个条件需要同时满足 Defaults to 0.5.
    
        """
        self.item = item
        self.start_time = start_time
        self.start_circle_num = start_circle_num
        self.waiting_num = waiting_num
        self.waiting_time = waiting_time
        self.remove_tag = False

    def try_recover_item(self,db:Database,item_frame_new:list,circle_num:int,recover_distance:float=recover_distance) ->None:
        """尝试恢复物体

        Args:
            db (Database): database
            item_frame_new (list): frame中新增list
            circle_num (int): 当前循环次数
            stable_range (float, optional): 恢复判定参数. Defaults to 0.01.
        """
        item_db_0 = db.db_read_exist()
        a,b,c,d = search_stable_item([self.item],item_frame_new,stable_distance)
        if(a.__len__()==0): #在该item附近寻找frame里是否有对应物体，如果没有说明不在
            self.remove_tag = True
        else:
            if((time.time()-self.start_time)>=self.waiting_time and (circle_num-self.start_circle_num)>=self.waiting_num):
                remove_list = db.db_read_remove()
                e,f,g,h = search_recover_stable_item([self.item],remove_list,recover_distance)
                if(e.__len__()==0):
                    self.remove_tag = True
                else: # e 待恢复frame元素 f 待恢复db中status=0元素
                    item_move_correlate = [-1]
                    temp_list = []
                    for i in item_db_0:
                        dist = item_distance(self.item,i)
                        if(item_if_close_correlate(self.item,i,dist)):
                            item_move_correlate = [i.uid]
                            temp_list.append([i.uid,dist])
                    if(len(temp_list)!=0):
                        temp_list.sort(key = lambda x:x[1])
                        item_move_correlate = [temp_list[0][0]]
                                        
                    db.db_recover([f[0].uid],b,item_move_correlate)

                    self.remove_tag = True

    def try_add_item(self,db:Database,item_frame_new:list,circle_num:int,add_range:list=add_range) ->None:
        """尝试增加物体

        Args:
            db (Database): database
            item_db_0 (list): db中所有存在物体
            item_frame_new (list): frame中新增list
            circle_num (int): _description_
            add_range (list, optional): _description_. Defaults to [0,0.15].
        """
        item_db_0 = db.db_read_exist()
        a,b,c,d,z = search_same_cls_in_range(item_db_0,[self.item],item_frame_new,add_range) #物品可能是移动着进入的
        if(a.__len__()==0):
            self.remove_tag = True
        else: #若物体持续在frame中新增list中存在
            if((time.time()-self.start_time)>=self.waiting_time and (circle_num-self.start_circle_num)>=self.waiting_num):
                e,f,g,h,z = search_same_cls_overlapping(item_db_0,[self.item],item_db_0) #db中是否存在相同cls高覆盖率的物体
                if(e.__len__()==0): #若不存在
                    db.db_save(b)
                    
                """else:
                    if(item_if_near_edge(self.item)):
                        db.db_save(b)
                    else:
                        db.db_appear(b,z)"""

                self.remove_tag = True

    def can_remove_this_objects(self) ->bool:
        if(self.remove_tag):
            return(True)
        else:
            return(False)

class RemoveItemWaiting:
        
    def __init__(self,item:Item,start_circle_num:int,start_time:float=time.time(),waiting_num:int = 3,waiting_time:float = 1):
        self.item = item
        self.start_time = start_time
        self.start_circle_num = start_circle_num
        self.waiting_num = waiting_num
        self.waiting_time = waiting_time
        self.remove_tag = False

    def try_remove_item(self,db:Database,item_frame_new:list,circle_num:int,remove_range:list=[0,0.5]) ->None:#内有可调节参数
        """尝试删除物体

        Args:
            db (Database): database
            item_db_0 (list): 数据库列表
            item_frame_new (list): _description_
            circle_num (int): _description_
            remove_range (list, optional): 此范围内认为是在移动不删除. Defaults to [0,0.5].
        """
        item_db_0 = db.db_read_exist()
        a,b,c,d,z = search_same_cls_in_range(item_db_0,[self.item],item_frame_new,remove_range)
        if(a.__len__()!=0):
            db.db_update(a,b,z)
            self.remove_tag = True
        else:#如果该物体一直不存在
            if((time.time()-self.start_time)>=self.waiting_time and (circle_num-self.start_circle_num)>=self.waiting_num):
                e,f,g,h,z = search_in_range(item_db_0,[self.item],item_db_0,range=[0,0.2])
                if(item_if_near_edge(self.item)):
                    db.db_remove([self.item])
                else:
                    if(len(z)!=0):
                        db.db_cover([self.item],z)
                    else:
                        db.db_remove([self.item])
                self.remove_tag = True

    def can_remove_this_objects(self) ->bool:
        if(self.remove_tag):
            return(True)
        else:
            return(False)


# %%
def get_pic_and_use_YOLO(frame,model,time_YOLO,time_list):
    """_summary_

    Args:
        time_YOLO (_type_): YOLO时间
        time_camera (_type_): camera时间
        time_list (_type_): 
        circle_num (_type_): _description_

    Returns:
        _type_: _description_
    """


    # 使用YOLO模型
    time_YOLO.get_init_time()
    item_frame_0 = frame2item_array(frame,model=model) # 拍摄照片中的物品及其参数
    time_list[2] = time_YOLO.get_delta_time()
    return(item_frame_0,time_list)
      
def analyse_data(db,item_frame_0,add_item_waiting_list,remove_item_waiting_list,time_db,circle_num,time_list):
    time_db.get_init_time()
    item_db_0 = db.db_read_exist() # 读取数据库中已存在的物品
    stable = search_stable_item(item_db_0,item_frame_0,0.01)
    move = search_same_cls_in_range(item_db_0,stable[2],stable[3],[0.0,0.2])
    db.db_update(move[0],move[1],move[4])
    for num1 in move[3]:
        add_item_waiting_list.append(AddItemWaiting(num1,circle_num))
    for num2 in move[2]:
        remove_item_waiting_list.append(RemoveItemWaiting(num2,circle_num))

    newlist_1 = list(add_item_waiting_list)
    newlist_2 = list(remove_item_waiting_list)

    for i in add_item_waiting_list:
        i.try_recover_item(db,move[3],circle_num)
        if(i.can_remove_this_objects()):
            if(i in newlist_1):
                newlist_1.remove(i)
        i.try_add_item(db,move[3],circle_num)
        if(i.can_remove_this_objects()):
            if(i in newlist_1):
                newlist_1.remove(i)

    for i in remove_item_waiting_list:
        i.try_remove_item(db,move[3],circle_num)

        if(i.can_remove_this_objects()):
            if(i in newlist_2):
                newlist_2.remove(i)
    time_list[3] = time_db.get_delta_time()
    return(newlist_1,newlist_2)

# %%
if __name__=='__main__':

    

    circle_count = Uid(0)
    model = YOLO(model_name)
    names = model.names
    db = Database(names,db_name = db_name,db2_name = db2_name,console_output=True)
    
    time_fps = TimeRunning('fps')
    time_camera = TimeRunning('camera')
    time_YOLO = TimeRunning('YOLO')
    time_db = TimeRunning('database')
    time_list = [0,0,0,0]

    camera = cv2.VideoCapture(0)
    
    add_item_waiting_list = []#新增物品等待区
    remove_item_waiting_list = []#移出物品等待区


    while(True):
        time_fps.get_init_time()
        circle_num = circle_count.getuid()


        time_camera.get_init_time()
        ret,frame = camera.read()
        if ret == False:
            print('摄像头连接失败！')
            cv2.destroyAllWindows()
            break
        time_list[1] = time_camera.get_delta_time()

        
        item_frame_0,timelist = get_pic_and_use_YOLO(frame,model,time_YOLO,time_list)
        newlist_1,newlist_2 = analyse_data(db,item_frame_0,add_item_waiting_list,remove_item_waiting_list,time_db,circle_num,time_list)

        add_item_waiting_list = newlist_1
        remove_item_waiting_list = newlist_2

        time_list[0] = time_fps.get_delta_time()
        cv2.namedWindow('Item tracker')
        cv2.imshow('Item trackr',draw_img(frame,db.db_read_exist(),names,time_list))
        cv2.waitKey(1)
        if (cv2.waitKey(1)==27):
            cv2.destroyAllWindows()
            print('Pressed esc and will exit!')
            break
        

# %%


# %%



