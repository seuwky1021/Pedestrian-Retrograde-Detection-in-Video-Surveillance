import cv2


import numpy as np
import time
camera = cv2.VideoCapture(r"C:\Users\86137\Desktop\20220825采集\普通来回\逆\1.mp4") # 参数0表示第一个摄像头
# videoWriter = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 12, (500,300))
# out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 12, (238, 158))

cv2.namedWindow("result",0)
i=1
while True:
    # 读取视频流
    # time.sleep(0.03)
    grabbed, img = camera.read()
    img=cv2.resize(img,(238,158))
    cv2.imshow('result', img)

    if i<10:
        a="00"+str(i)
    elif i<100:
        a="0"+str(i)
    else:
        a=str(i)

    cv2.imwrite("mydataset/Test/Test012/"+a+".tif",img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):    # 按'q'健退出循环
        break
    i+=1

    # videoWriter.write(img)

# 释放资源并关闭窗口
camera.release()
cv2.destroyAllWindows()
