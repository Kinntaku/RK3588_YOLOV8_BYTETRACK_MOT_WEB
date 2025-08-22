OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.5, 0.9,1024


TARGET_CLASSES ={"person","dog","cat"} #填写自己需要追踪的类别
CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse", "remote", "keyboard", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")


model = "./rknnModel/ch_200_ReLU_best_1024.rknn" #使用crowdhuman数据集训练200次的1024输入大小，relu作为激活函数的模型
#./rknnModel/ch_best_180_1024.rknn ：使用crowdhuman训练的1024输入大小的模型，
#./rknnModel/ch_best_180_640.rknn ：使用crowdhuman训练的640输入大小的模型（使用时记得修改IMG_SIZE），
#./rknnModel/yolov8s.rknn ：yolov8s默认模型（使用时记得修改IMG_SIZE）
