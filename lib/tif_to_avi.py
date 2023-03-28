import cv2
import numpy as np
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('I','4','2','0')
videoWriter = cv2.VideoWriter("test8二逆一顺.avi", fourcc, 30, (500,300))
cv2.namedWindow("Image")

score=[]
with open('../Colab结果/score-8二逆一顺.txt','r') as f:
    for ann in f.readlines():
        ann = ann.strip('\n')  # 去除文本中的换行符
        score.append(float(ann))

print(score)
i=81
while(True):
    if i<10:
        a="00"+str(i)
    elif i>=10 and i<=99:
        a="0"+str(i)
    elif i<=320:
        a=str(i)
    else:
        break

    a="../mydataset/Test/Test008erniyishun/"+a+".tif"
    img=cv2.imread(a)

    # ------------------------------
    # 进行处理，添加字幕
    img=cv2.resize(img,(500,300),interpolation=cv2.INTER_AREA)
    # threshold=np.mean(score)
    threshold=0.95

    if score[i]<threshold and i>140:
        cv2.putText(img, "Retrograde Or Not: YES", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        # print(3)
        cv2.putText(img, "Retrograde Or Not: NO", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # -------------------------------

    videoWriter.write(img)
    print(a)
    cv2.imshow("Image",img)
    i+=1
# img=cv2.imread("./img32/001.tif")
# cv2.imshow("Image",img)
videoWriter.release()
cv2.waitKey(0)
#释放窗口
cv2.destroyAllWindows()
