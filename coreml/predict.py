import coremltools
import sys

sys.path.append('../keras/')
import mcz_input

(X_test, y_test)= mcz_input.read_data('../deeplearning/predict.txt')
#(X_test, y_test)= mcz_input.read_data('../deeplearning/train.txt')
X_test = X_test.astype('float32')
X_test /= 255

#print(X_test)
#print(y_test)
print('loading..')
model = coremltools.models.MLModel('Momomind.mlmodel')
print('predicting..')
res = model.predict(X_test)
print(res)
