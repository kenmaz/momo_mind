path = '../keras/model.h5'

import coremltools
coreml_model = coremltools.converters.keras.convert(path,
        input_names = 'image',
        image_input_names = 'image',
        #is_bgr = True,
        image_scale = 0.00392156863,
        class_labels = 'labels.txt')

coreml_model.save('Momomind.mlmodel')

