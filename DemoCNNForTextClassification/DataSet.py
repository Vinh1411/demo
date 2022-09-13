import gensim
import os
import pickle
from pyvi import ViTokenizer

current_directory=os.getcwd()

path_train=current_directory+"\\DemoCNNForTextClassification\\Train"
path_test=current_directory+"\\DemoCNNForTextClassification\\Test"

X_train=[]
Y_train=[]

X_test=[]
Y_test=[]

def readData(path, X, Y, labels):
    for label in labels:
        path_label=path+"\\"+label
        files=os.listdir(path_label)
        for file in files:
            with open(path_label+"\\"+file, 'r', encoding="UTF-16") as f:
                data=f.read()
                #Loại bỏ ký tự đặc biệt và tách thành dữ liệu tành từng từ
                data=gensim.utils.simple_preprocess(data)
                data=' '.join(data)
                #Xử lý văn bản (kết hợp các từ lại để đảm bảo nghĩa)
                data=ViTokenizer.tokenize(str(data))
                X.append(data)
                Y.append(label)

readData(path_train, X_train, Y_train, os.listdir(path_train))
readData(path_test, X_test, Y_test, os.listdir(path_test))

#Lưu dữ liệu xử lý lần 1 vào file định dạng pkl
pickle.dump(X_train, open('DemoCNNForTextClassification/Pre_data_1/Train/X_train.pkl', 'wb'))
pickle.dump(Y_train, open('DemoCNNForTextClassification/Pre_data_1/Train/Y_train.pkl', 'wb'))

pickle.dump(X_test, open('DemoCNNForTextClassification/Pre_data_1/Test/X_test.pkl', 'wb'))
pickle.dump(Y_test, open('DemoCNNForTextClassification/Pre_data_1/Test/Y_test.pkl', 'wb'))
