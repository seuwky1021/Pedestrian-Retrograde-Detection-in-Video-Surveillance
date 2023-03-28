for i in range(10):
    with open('test.txt','a') as f:
        str1=str(i)+str('\n')
        f.write(str1)
