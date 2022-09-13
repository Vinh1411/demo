import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras_preprocessing.sequence import pad_sequences
from sklearn import preprocessing

#Đọc dữ liệu train đã được chuẩn hóa
X_train=pickle.load(open('DemoCNNForTextClassification/Pre_data_1/Train/X_train.pkl', 'rb'))
Y_train=pickle.load(open('DemoCNNForTextClassification/Pre_data_1/Train/Y_train.pkl', 'rb'))

#Đọc dữ liệu test đã được chuẩn hóa
X_test=pickle.load(open("DemoCNNForTextClassification/Pre_data_1/Test/X_test.pkl", 'rb'))
Y_test=pickle.load(open("DemoCNNForTextClassification/Pre_data_1/Test/Y_test.pkl", 'rb'))

#Mã hóa nhãn dữ liệu
encoder=preprocessing.LabelEncoder()
encoder.fit(Y_train)
Y_train=encoder.fit_transform(Y_train)
Y_test=encoder.fit_transform(Y_test)

#Mã hóa văn bản với mỗi phần tử là 1 giá trị số nguyên khác nhau, chu y no khong su dung phan tu 0
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
pre_X_train=tokenizer.texts_to_sequences(X_train)
pre_X_test=tokenizer.texts_to_sequences(X_test)

#Tính thêm cả 0 nữa nếu không thì khi đưa vào nó sẽ xuất hiện là nằm ngoài index
vocal_size=len(tokenizer.word_index)+1
maxlen=16422
#Chuẩn hóa sao cho mỗi văn bản đầu vào giống nhau.
pre_X_train = pad_sequences(pre_X_train, padding='post', maxlen=maxlen)
pre_X_test = pad_sequences(pre_X_test, padding='post', maxlen=maxlen)

#Tạo một mô hình tuần tự
model=Sequential()
model.add(layers.Embedding(vocal_size, 200, input_length=maxlen))
#Đưa vào 1 lớp tích chập với 128 filter 1 chiều kích thước 5*200 --> feature map. 
model.add(layers.Conv1D(128, 5, activation='relu'))
#Giảm kích thước
model.add(layers.GlobalMaxPooling1D())
#Chồng 1 lớp FFNN
model.add(layers.Dense(10, activation='relu'))
#Output
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#print(model.summary())
print(vocal_size)
#model.fit(pre_X_train, Y_train, epochs=10, verbose=False, validation_data=(pre_X_test, Y_test), batch_size=10)

#pickle.dump(model, open("DemoCNNForTextClassification/Model/model.pkl", 'wb'))
#model.save('myModel.h5')
#pickle.dump(pre_X_train, open("DemoCNNForTextClassification/Pre_data_2/Train/X_train.pkl", "wb"))
#pickle.dump(pre_X_test, open("DemoCNNForTextClassification/Pre_data_2/Test/X_test.pkl", "wb"))
'''
tfidfVector=TfidfVectorizer(analyzer='word', max_features=30000)
X_train_tfidf=tfidfVector.fit(X_train)
X_train_tfidf=tfidfVector.transform(X_train)
X_test_tfidf=tfidfVector.transform(X_test)
pickle.dump(X_train_tfidf, open("DemoCNNForTextClassification/Pre_data_2/Train/X_train_tfidf.pkl", "wb"))
pickle.dump(X_test_tfidf, open("DemoCNNForTextClassification/Pre_data_2/Test/X_test_tfidf.pkl", "wb"))
'''