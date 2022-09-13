import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

X_train=pickle.load(open("DemoCNNForTextClassification/Pre_data_2/Train/X_train.pkl", "rb"))
Y_train=pickle.load(open("DemoCNNForTextClassification/Pre_data_1/Train/Y_train.pkl", "rb"))

X_test=pickle.load(open("DemoCNNForTextClassification/Pre_data_2/Test/X_test.pkl", "rb"))
Y_test=pickle.load(open("DemoCNNForTextClassification/Pre_data_1/Test/Y_test.pkl", "rb"))

#model=pickle.load(open("DemoCNNForTextClassification/Model/model.pkl", 'rb'))
model = load_model('myModel.h5')

encode=LabelEncoder()
encode.fit(Y_train)
Y_train=encode.fit_transform(Y_train)
Y_test=encode.fit_transform(Y_test)

loss, accuracy=model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy=model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))