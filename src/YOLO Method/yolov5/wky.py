import cv2
import numpy as np
import math
import time


vc = cv2.VideoCapture(r"D:\Steven\yolov5\yolov5\data\videos\reverse.mp4")  # 读入视频文件

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 确定解码器
out = cv2.VideoWriter('save_elements1.mp4', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 30.0, (640, 360))

fps = vc.get(cv2.CAP_PROP_FPS)
print(fps)
vfps = 1000 / fps
#vc = cv2.VideoCapture("E:\大三下\科研与工程实践\正常.mp4")  # 读入视频文件
#vc = cv2.VideoCapture("E:\大三下\科研与工程实践\背景.mp4")  # 读入视频文件
#vc = cv2.VideoCapture("./normal.mp4")  #路径中不带中文，否则后面会报错

# ShiTomasi corner detection的参数
feature_params = dict(maxCorners=1000,
                      qualityLevel=0.15,
                      minDistance=7,
                      blockSize=7)
# 光流法参数
# maxLevel 未使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))

rval, firstFrame= vc.read()

#frameToStart = 10        #捕获的是第几帧
#vc.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
#ret,firstFrame = vc.retrieve() # 此时返回的frame便是第25帧图像
global shun
global ni
shun=0
ni=0

# 判断两个向量的夹角
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def retrograde(old_x,old_y,new_x,new_y):#用来判断是否逆行的函数
    global shun, ni
    Vx = new_x-old_x #横 向右为正
    Vy = new_y-old_y #纵 向下为正
    temp=[0,0,Vx,Vy]  #要检测的方向的向量，前两个数字代表起始点，后两个是向量的终点
    # shunxing=[0,0,1,-1]  #规定的顺行的方向，向右为x的正方向，向下为y的正方向，所以-1代表向上为顺行方向
    shunxing=[0,0,0,-1]  #规定的顺行的方向，向右为x的正方向，向下为y的正方向，所以-1代表向上为顺行方向
    # nixing=[0,0,-1,1]   #同上，向下规定为逆行
    nixing=[0,0,0,1]   #同上，向下规定为逆行
    Vmin=0
    Vmax=1#有待修改
    if math.sqrt(Vx**2+Vy**2)<=Vmin and math.sqrt(Vx**2+Vy**2)>Vmax:
        print("有跟踪错误的点")
    theta_margin = 45 #顺行的余量为45°
    theta_ni_margin=45 #和逆行方向夹角在5°以内的规定为逆行
    # theta1=90       #顺行方向
    # theta2 = -90    #逆行方向

    # angle函数，判断两个向量的夹角（输出的是角度制度，范围在0-180之间）

    if angle(temp,shunxing)<theta_margin:
        # print("顺行:"+str(angle(temp,shunxing)))
        shun += 1
    if angle(temp,nixing)<theta_ni_margin:
        print(angle(temp, nixing))
        # print("----------------有人逆行----------------:"+str(angle(temp,nixing)))
        ni +=1
    if math.atan2(Vx, Vy) / math.pi * 180 ==0:
        print("角点未移动")
    return 0

def getlen(a, b):#a b 是两个角点的集合
    list = []
    """
    if len(a)!=len(b):
        print("两个角点集合中角点数不同")
        return list
    """
    for i in range(len(a)):
        list.append(math.sqrt((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2))
    return list

def Avg_vel(a,b,NumOfFrames): #NumOfFrames表示ab之间间隔了多少帧，大于等于1
    list=getlen(a,b)
    Total_dist=0
    for i in range(len(list)):
        Total_dist += list[i]
    print((Total_dist/len(list))/NumOfFrames)
    return (Total_dist/len(list))/NumOfFrames

def Sin_vel(a,b,NumOfFrames):
    list = getlen(a, b)
    del_list=[]
    for i in range(len(list)):
        print(list[i]/NumOfFrames)
        if list[i]/NumOfFrames<0.1:
            del_list.append(i)
    return del_list#返回静止不动的点的集合

firstFrame = cv2.resize(firstFrame, (640, 360), interpolation=cv2.INTER_CUBIC)
mask = np.zeros_like(firstFrame)  # 为绘制创建掩码图片
gray_firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)  # 灰度化
firstFrame = cv2.GaussianBlur(gray_firstFrame, (21, 21), 0)  # 高斯模糊，用于去噪
prveFrame = firstFrame.copy()

prospect_frame = 1 #提取到的前景的第一帧
# 遍历视频的每一帧


index=0
array_ni=[]
for i in range(9999):
    array_ni.append(0)
while True:
    # time.sleep(vfps/1000)
    (ret, frame) = vc.read()#说明：按帧读取视频，返回值ret是布尔型，正确读取则返回True，读取失败或读取视频结尾则会返回False。frame为每一帧的图像，这里图像是三维矩阵，即frame.shape = (640,480,3)，读取的图像为BGR格式。

    # print("------------",frame.shape)
    # 如果没有获取到数据，则结束循环
    if not ret:
        break

    # 对获取到的数据进行预处理
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

    # 计算当前帧与上一帧的差别
    frameDiff = cv2.absdiff(prveFrame, gray_frame)#差分图像

    # 忽略较小的差别
    retVal, thresh = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)

    # 对阈值图像进行填充补洞
    thresh = cv2.dilate(thresh, None, iterations=2)

    # image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    area = np.zeros(frame.shape[0:2], dtype="uint8")#每一帧提取到的大致前景区域
    for contour in contours:
        # if contour is too small, just ignore it
        if cv2.contourArea(contour) < 50:  # 面积阈值50
            continue

        # 计算最小外接矩形（非旋转）
        (x, y, w, h) = cv2.boundingRect(contour)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(area, (x, y), (x + w, y + h), 255, -1)
    gap=100 #每隔100帧重新进行一次角点检测
    if prospect_frame == 1:  #第一帧图像虚化严重，放弃第一帧，从第二帧开始检测
        prveFrame = gray_frame.copy()
        prospect_frame += 1
        continue
    elif prospect_frame ==2 or (prospect_frame-2)%gap ==0:
        p0 = cv2.goodFeaturesToTrack(gray_frame, mask=area, **feature_params)  #角点检测
        # print(p0,"-------------")
        Origin_Corner = p0.copy().reshape(len(p0),2)
        del_list = []
        """for i in p0:    #p0此时是第一帧的角点集合，p1是下一帧的角点的集合
            x, y = i.ravel()
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), 2)"""

    else:
        #status ：输出状态向量（无符号字符）;如果找到相应特征的流，则向量的每个元素设置为1，否则设置为0。
        p1, st, err = cv2.calcOpticalFlowPyrLK(prveFrame, gray_frame, p0, None, **lk_params)  # 用于获得光流检测后的角点位置
        # print(p1,"-------------")
        # 选择good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if prospect_frame==10 or (prospect_frame-12)%100 ==0:#角点重新检测后10帧找到几乎静止不动的点
            Avg_vel(p0.reshape(len(p0), 2), Origin_Corner, 10)
            del_list=Sin_vel(p0.reshape(len(p0),2),Origin_Corner,10)
            temp1 = []
            temp2 = []
            for j in range(len(good_old)):
                if j not in del_list:
                    temp1.append(good_old[j])
                    temp2.append(good_new[j])
            good_old = np.array(temp1)
            good_new = np.array(temp2)

        # 绘制跟踪框
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)#蒙版上的图案可以在图像中一直显示
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)#在原图上的圆形会一直更新
            retrograde(c,d,a,b)
        print("顺行点数",shun)
        print("逆行点数",ni)
        array_ni[index]=ni
        img = cv2.add(frame, mask)

        #判断逆行的阈值
        threshhold=4
        #综合前后5帧判断，实现精检测
        framenum=5
        print("index"+str(index))
        print(array_ni)
        print(len(array_ni))
        if index<framenum:
            if ni<threshhold:
                cv2.putText(img,"Retrograde Or Not: NO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(img,"Retrograde Or Not: YES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # 判断之前10帧内是否ni都小于threshold，若是，temp就是True
            temp=True
            for j in range(framenum):
                temp=temp and (array_ni[index-j]<threshhold)
            if temp == True:
                for k in range(framenum-2):
                    temp=temp and (array_ni[index-k-1])
                for k in range(framenum-2):
                    temp=temp and (array_ni[index-k-2])
                if temp == True:
                    cv2.putText(img, "Retrograde Or Not: NO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    cv2.putText(img, "Retrograde Or Not: YES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "Retrograde Or Not: YES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        shun=0
        ni=0

        out.write(img)
        cv2.imshow('frame', img)
        #cv2.imshow('area', area)


        p0 = good_new.reshape(-1, 1, 2)
    prveFrame = gray_frame.copy()
    prospect_frame += 1

    index+=1

    # 处理按键效果
    key = cv2.waitKey(60) & 0xff
    if key == 27:  # 按下ESC时，退出
        break
    elif key == ord(' '):  # 按下空格键时，暂停
        cv2.waitKey(0)

    # cv2.waitKey(0)
cv2.destroyAllWindows()
vc.release()
print("shun",shun)
print("ni",ni)
