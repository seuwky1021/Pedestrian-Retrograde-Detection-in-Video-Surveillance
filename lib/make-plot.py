import cv2
import numpy as np


score=[]
i=1
with open('../Colab结果/score-7一人折回.txt','r') as f:
    for ann in f.readlines():
        if i>30:
            ann = ann.strip('\n')  # 去除文本中的换行符
            score.append(float(ann))
        i+=1

print(score)
import matplotlib.pyplot as plt
# plt.ylim((0.70, 1))
plt.plot(score)
plt.ylabel('regularity score')
# plt.show()
plt.savefig("7.jpg")
