import os

Name_label = [] 
path = './data/face/'   
dir = os.listdir(path) 
label = 0

#ghi dữ liệu
with open('./data/train.txt','w') as f:
    for name in dir:
        Name_label.append(name)
        print(Name_label[label])
        after_generate = os.listdir(path +'\\'+ name)
        for image in after_generate:
            if image.endswith(".png"):
                f.write(image + ";" + str(label)+ "\n")
        label += 1
