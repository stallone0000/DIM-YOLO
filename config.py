import numpy as np
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


