from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from RADNet import *
#超参数
IMAGE_SIZE = 128
LR = 0.0001  
epochs = 600
batch_size = 4 #大数据集采用大点的批处理
input_sizes = (IMAGE_SIZE,IMAGE_SIZE,3)
classes = 2
# =============================================================================

'''
选择训练集
'''
data = r"C:\Users\Administrator.DESKTOP-3NIAR4J\Desktop\folder\model\data"



'''
训练模型时读取npy文件
'''
def readNpy():
    train_image = np.load(data+"/imagesDataset_.npy")
    train_GT    = np.load(data+"/labelsDataset_.npy")
    val_image = np.load(data+"/val_imagesDataset_.npy")
    val_GT    = np.load(data+"/val_labelsDataset_.npy")
    edge_GT = np.load(data+"/edgeDataset_.npy")
    val_edge_GT    = np.load(data+"/val_edgeDataset_.npy")
    return train_image,train_GT,val_image ,val_GT,edge_GT ,val_edge_GT    




'''
选择训练模型和训练集
'''
# model_list = ['unet','pspnet','segnet','deeplabv3+']
model_list = ['RADNet']
for net in model_list:
    if net=='RADNet':
          model =RADNet(2, (IMAGE_SIZE,IMAGE_SIZE,3), epochs, batch_size, LR, Falg_summary=True, Falg_plot_model=False)
    else:
        print('no model')
   # net = net +'-test4'
    '''
    保存模型文件
    '''
    def mkSaveDir(mynet):
        TheTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        savePath = './logs/' + str(mynet) + '-' + str(TheTime)
        if not os.path.exists(savePath): 
            os.makedirs(savePath)
        return savePath
    
    '''
    使用保存点
    '''
    savePath = mkSaveDir(net)
    checkpointPath= savePath + "/"+net+"-1"+"-{epoch:03d}-{val_seg_out1_accuracy:.4f}-{seg_out1_accuracy:.4f}.hdf5"
    checkpoint = ModelCheckpoint(checkpointPath, monitor='seg_out_accuracy', verbose=1,
                                 save_best_only=False, save_weights_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=savePath, histogram_freq=0)
    callback_lists = [tensorboard, checkpoint]  
    '''
    读取训练集，开始训练
    '''
    train_image,train_GT,val_image ,val_GT,edge_GT ,val_edge_GT   = readNpy()
    # train_GT1 = to_categorical(train_GT)
    # edge_GT1 = to_categorical(edge_GT)
    # val_GT1 = to_categorical(val_GT)
    History = model.fit(train_image, [train_GT,edge_GT], batch_size=batch_size, validation_data=(val_image, [val_GT,val_edge_GT]),
                        epochs=epochs, verbose=1, shuffle=True, class_weight=None, callbacks=callback_lists)
    with open(savePath + '/log_128.txt','w') as f:
        f.write(str(History.history))
    model.save_weights(savePath + '/save_weights.h5')
    



