
import os
import glob
import h5py
import keras
import numpy as np
from Name import *
from PIL import Image
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils.utils import get_random_data
#công cụ gán nhãn đối chiếu khớp với tham số của số lượng nhãn
image_w = 300 #chiều rộng của hình ảnh
image_h = 300 #chiều cao của hình ảnh
num_class = 40 #số lượng nhãn

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

#2
def generate_arrays_from_file(lines,batch_size,train):
    # tính tổng độ dài
    n = len(lines)
    i = 0
    while 1:
        X_train = []    #300x300x3
        Y_train = []    #label
        # lấy một lô dữ liệu có kích thước batch_size
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            label_name = (lines[i].split(';')[1]).strip('\n')
            file_name = str(Name.get(int(label_name)))
            # đọc hình ảnh từ tệp
            img = Image.open(r".\data\face" +"\\"+ file_name +"\\"+ name)
            if train == True:
                img = np.array(get_random_data(img,[image_h,image_w]),dtype = np.float64)
            else:
                img = np.array(letterbox_image(img,[image_h,image_w]),dtype = np.float64)
            X_train.append(img)
            Y_train.append(label_name)
            # sau khi đọc xong một chu kỳ, bắt đầu lại từ đầu
            i = (i+1) % n
        # xử lý hình ảnh
        X_train = preprocess_input(np.array(X_train).reshape(-1,image_h,image_w,3))
        #one-hot
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= num_class)   
        yield (X_train, Y_train)
#3 
#2. xây dựng mô hình
def MmNet(input_shape, output_shape):
    model = Sequential()  # xây dựng mô hình
    # tầng thứ nhất
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # tầng thứ hai
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # tầng thứ ba
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # tầng fully connected (FC)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # tầng fully connected (FC)
    model.add(Dense(output_shape, activation='softmax'))
    print("-----------Tóm tắt mô hình----------\n")  # tóm tắt mô hình
    model.summary()

    return model
#4
#3. đào tạo mô hình
def train(model, batch_size):

    model = model   #đọc mô hình
    #Định nghĩa cách lưu trữ, lưu mô hình sau mỗi ba epochs
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        period=3
    )
    #Cách giảm learning rate, nếu accuracy không tăng trong ba epochs liên tiếp thì giảm learning rate và tiếp tục đào tạo
    reduce_lr = ReduceLROnPlateau(
        monitor='accuracy',
        patience=3,
        verbose=1
    )
    #Nếu val_loss không giảm trong một thời gian dài, điều đó có nghĩa là mô hình đã được đào tạo đầy đủ và có thể dừng đào tạo
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )
    #độ mất mát chéo
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    #trực quan hóa Tesorboard
    tb_Tensorboard = TensorBoard(log_dir="./model", histogram_freq=1, write_grads=True)
    #bắt đầu huấn luyện
    
    history = model.fit(self, x = generate_arrays_from_file(lines[:num_train], batch_size, True),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
            validation_steps=max(1, num_val//batch_size),
            verbose = 1,
            epochs=3,
            initial_epoch=0,
            callbacks=[early_stopping, checkpoint_period1, reduce_lr])
    return history, model
    
#5
#4. tạo dữ liệu kiểm tra
def test_data(lines):
    # lấy tổng độ dài tập dữ liệu kiểm tra
    n = len(lines)
    i = 0   #bộ đếm
    # lấy danh sách dữ liệu kiểm tra và nhãn
    x_test = []    #Kích thước của dữ liệu kiểm tra
    y_test = []    #Kích thước của nhãn kiểm tra
    # Lặp qua các dữ liệu trong tập kiểm tra để thực hiện dự đoán và tính toán độ chính xác.
    for i in range(n):
        name = lines[i].split(';')[0]   #Tên của hình ảnh khuôn mặt xxx_xxx.png
        label_name = (lines[i].split(';')[1]).strip('\n')   #Nhãn số của khuôn mặt 0-39
        file_name = str(Name.get(int(label_name)))  #Đối ứng với tên của người str
        # Đọc hình ảnh từ tệp
        img = Image.open(r".\data\face" +"\\"+ file_name +"\\"+ name)
        img = np.array(letterbox_image(img,[image_h,image_w]),dtype = np.float64)
        x_test.append(img)
        y_test.append(label_name)
        # Sau khi đọc xong một chu kỳ, bắt đầu lại từ đầu
        i += 1  #tăng bộ đếm lên 1
    # Xử lý hình ảnh
    x_test = preprocess_input(np.array(x_test).reshape(-1,image_h,image_w,3))
    # Mã hóa one-hot
    y_test = np_utils.to_categorical(np.array(y_test),num_classes= num_class)   
    return x_test, y_test

#6
if __name__ == "__main__":
    #Đường dẫn lưu trữ cho mô hình được huấn luyện
    log_dir = "./logs/"
    #Dữ liệu đường dẫn khuôn mặt
    path = "./data/face/"
    # Mở tệp tin dữ liệu văn bản của bộ dữ liệu.
    with open(r".\data\train.txt","r") as f:
        lines = f.readlines()
        
    
    # Đảo ngược dữ liệu huấn luyện
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # Được chia thành 80% cho huấn luyện và 20% cho kiểm tra.
    num_val = int(len(lines)*0.2)   #
    num_train = len(lines) - num_val  #4715
    # #Dữ liệu khởi tạo
    # x_train, y_train, x_test, y_test = read_image(path, image_w, image_w) 


    #Định nghĩa tham số của mô hình

    input_shape = (image_w, image_h, 3) #Đầu vào
    output_shape = num_class    #Khởi tạo đầu ra của mô hình sẽ là một vectơ có số chiều bằng với số lượng nhãn (số người trong tập dữ liệu)
    #mô hình AlexNet bằng TensorFlow và keras
    model = MmNet(input_shape, output_shape)
    batch_size = 32
    try:
        model = load_model(log_dir + 'modal1.h5')
    except OSError:
        history, model = train(model,batch_size)
    #else:
        history, model = train(model,batch_size)
    model.save(log_dir + 'modal1.h5')    #lưu mô hình và trọng số đã được đào tạo
   
    #In kết quả dự đoán
    x_test, y_test = test_data(lines[num_train:])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Đánh giá kết quả dự đoán mô hình')
    print('Độ Sai Số:', score[0]*100)
    print('Độ Chính Xác:', score[1]*100)

