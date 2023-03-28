'''1.实时性问题
2.精检测的问题
3.人头检测的精确度'''

import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
import math

video_path = r'D:\Steven\yolov5\yolov5\data\videos\reverse.mp4'

#####yolo parameters
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp29/weights/best.pt',
                    help='model.pt path(s)')
parser.add_argument('--source', type=str, default=video_path, help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args()
print(opt)

######光流法 parameters
# ShiTomasi corner detection的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.2,#0.3
                      minDistance=7,
                      blockSize=7)
# 光流法参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


#如果需要在输出的结果图上绘制运动轨迹的话，需要生成mask，在mask上面绘制直线，再把mask和原图加起来形成最后的效果图
'''
vc = cv2.VideoCapture(video_path)  # 读入视频文件
rval, firstFrame= vc.read()
mask = np.zeros_like(firstFrame)  # 为绘制创建掩码图片
'''

####下面是光流法前进性一些变量初始化
color = np.random.randint(0, 255, (100, 3))#角点需要的颜色集合
prospect_frame = 1 #提取到的前景的第一帧
index=0
gap=10#每隔10帧进行重新角点检测
array_ni=[]
for i in range(9999):
    array_ni.append(0)

global shun
global ni
shun=ni=0

#####接下来是判断逆行用到的一些函数
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
    shunxing=[0,0,0,-1]  #规定的顺行的方向，向右为x的正方向，向下为y的正方向，所以-1代表向上为顺行方向
    nixing=[0,0,0,1]   #同上，向下规定为逆行
    Vmin=0
    Vmax=1#有待修改
    if math.sqrt(Vx**2+Vy**2)<=Vmin and math.sqrt(Vx**2+Vy**2)>Vmax:
        print("有跟踪错误的点")
    theta_margin = 45 #顺行的余量为45°
    theta_ni_margin=5 #和逆行方向夹角在5°以内的规定为逆行
    # theta1=90       #顺行方向
    # theta2 = -90    #逆行方向

    # angle函数，判断两个向量的夹角（输出的是角度制度，范围在0-180之间）

    if angle(temp,shunxing)<theta_margin:
        #print("顺行:"+str(angle(temp,shunxing)))
        shun += 1
    if angle(temp,nixing)<theta_ni_margin:
        #print(angle(temp, nixing))
        #print("----------------有人逆行----------------:"+str(angle(temp,nixing)))
        ni +=1
    #if math.atan2(Vx, Vy) / math.pi * 180 ==0:
        #print("角点未移动")
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
    #print((Total_dist/len(list))/NumOfFrames)
    return (Total_dist/len(list))/NumOfFrames

def Sin_vel(a,b,NumOfFrames):
    list = getlen(a, b)
    del_list=[]
    for i in range(len(list)):
        #print(list[i]/NumOfFrames)
        if list[i]/NumOfFrames<0.1:
            del_list.append(i)
    return del_list#返回静止不动的点的集合


####接下来是yolo进行的一些操作
source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://'))

# Directories
save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Initialize
set_logging()
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

# Set Dataloader
vid_path, vid_writer = None, None
if webcam:
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
t0 = time.time()

####下面是对视频的每一帧进行处理，即dataset里存放的是视频的每一帧
for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            save_list = []
            im0_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)  # 灰度化
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    save_list.append(torch.tensor(xyxy).view(1, 4).tolist()[0])
                    #plot_one_box(xyxy, im0, label=None, color=colors[int(cls)], line_thickness=3)
                    plot_one_box(xyxy, im0, label=None, color=(0,255,0), line_thickness=3)

            area = np.zeros(im0.shape[0:2], dtype="uint8")  # 每一帧提取到的大致前景区域
            for xyxy in save_list:
                cv2.rectangle(area, (int(xyxy[0])-20,int(xyxy[1])-20), (int(xyxy[2])+20, int(xyxy[3])+20), 255, -1)#这里把头变大了
            if prospect_frame == 1 or (prospect_frame-1)%gap ==0:
                p0 = cv2.goodFeaturesToTrack(im0_gray, mask=area, **feature_params)  # 角点检测
                '''
                if p0==[]:##防止某一帧没有检测出角点的情况,把角点赋值成人头框中心点
                    trans=[]
                    for i in save_list:
                        trans.append([(i[0] + i[2]) / 2, (i[1] + i[3]) / 2])
                    p0=np.array(trans).reshape(-1, 1, 2)
                '''
                Origin_Corner = p0.copy().reshape(len(p0), 2)
                del_list = []
            else:
                # status ：输出状态向量（无符号字符）;如果找到相应特征的流，则向量的每个元素设置为1，否则设置为0。
                if p0.tolist() == []:##防止某一帧没有检测出角点的情况
                    prveFrame = im0_gray.copy()
                    prospect_frame += 1
                    index += 1
                    continue
                p1, st, err = cv2.calcOpticalFlowPyrLK(prveFrame, im0_gray, p0, None, **lk_params)  # 用于获得光流检测后的角点位置
                # 选择good points

                good_new = p1[st == 1]
                good_old = p0[st == 1]

                '''此部分代码为了减去未移动的点，但当gap很小（=3）时，会把good_new good_old里面的点全部删掉，从而error
                    问题可能在于判断点是否移动的条件出现错误，第23帧认为所有的点都没有移动，所以把所有的角点全部都删除掉了
                    
                if prospect_frame == 5 or (prospect_frame - 5) % gap == 0:  # 角点重新检测后10帧找到几乎静止不动的点
                    Avg_vel(p0.reshape(len(p0), 2), Origin_Corner, gap)
                    del_list = Sin_vel(p0.reshape(len(p0), 2), Origin_Corner, gap)
                    temp1 = []
                    temp2 = []
                    for j in range(len(good_old)):
                        if j not in del_list:
                            temp1.append(good_old[j])
                            temp2.append(good_new[j])
                    good_old = np.array(temp1)
                    good_new = np.array(temp2)
                '''
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    #mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)#蒙版上的图案可以在图像中一直显示
                    #im0 = cv2.circle(im0, (int(a), int(b)), 5, color[i].tolist(), -1)  # 在原图上的圆形会一直更新
                    im0 = cv2.circle(im0, (int(a), int(b)), 5, (255,0,0), -1)  # 在原图上的圆形会一直更新

                    retrograde(c, d, a, b)
                print("顺行点数", shun,'逆行点数',ni)

                #####接下来是精检测部分
                # 判断逆行的阈值
                array_ni[index] = ni
                threshhold = 3 #每一帧逆行的角点数量少于threshold就判断存在逆行
                # 综合前后5帧判断，实现精检测
                framenum = 5
                #print("index" + str(index))
                #print(array_ni)
                #print(len(array_ni))
                if index < framenum:
                    if ni < threshhold:
                        cv2.putText(im0, "Retrograde Or Not: NO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    else:
                        cv2.putText(im0, "Retrograde Or Not: YES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    3)
                else:
                    # 判断之前10帧内是否ni都小于threshold，若是，temp就是True
                    temp = True
                    for j in range(framenum):
                        temp = temp and (array_ni[index - j] < threshhold)
                    if temp == True:
                        for k in range(framenum - 2):
                            temp = temp and (array_ni[index - k - 1]< threshhold)
                        for k in range(framenum - 2):
                            temp = temp and (array_ni[index - k - 2]< threshhold)
                        if temp == True:
                            cv2.putText(im0, "Retrograde Or Not: NO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 3)
                        else:
                            cv2.putText(im0, "Retrograde Or Not: YES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 3)
                    else:
                        cv2.putText(im0, "Retrograde Or Not: YES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    3)
                ##########################
                shun = 0
                ni = 0
                #im0 = cv2.add(im0, mask)#在原图上添加直跟踪轨迹（直线）
                p0 = good_new.reshape(-1, 1, 2)
            prveFrame = im0_gray.copy()
            prospect_frame += 1
            index += 1
        # Print time (inference + NMS)
        #print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

if save_txt or save_img:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")

#print(f'Done. ({time.time() - t0:.3f}s)')


'''

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
'''