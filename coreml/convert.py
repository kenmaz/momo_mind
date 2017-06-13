#from keras.models import load_model
path = '/Users/kentaro.matsumae/Projects/momo_mind/keras/model.h5'
#model = load_model(path)

import coremltools
coreml_model = coremltools.converters.keras.convert(path, input_names = 'image', image_input_names = 'image', class_labels = 'labels.txt')
coreml_model.save('cifar10.mlmodel')

