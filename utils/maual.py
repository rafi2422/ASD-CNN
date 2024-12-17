
image_gen = ImageDataGenerator(rotation_range=20, 
                               width_shift_range=0.10, 
                               height_shift_range=0.10, 
                               rescale=1/255, 
                               shear_range=0.1, 
                               zoom_range=0.1, 
                               horizontal_flip=True, 
                               fill_mode='nearest'
                              )

image_gen.flow_from_directory(my_data_dir)

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=2)

model = load_model('/kaggle/input/trained/fruit.h5')
from tensorflow.keras.preprocessing import image
carambola1 = carambola+'/'+os.listdir(carambola)[15]
my_image = image.load_img(carambola1,target_size=image_shape)
my_image = image.img_to_array(my_image)
my_image = np.expand_dims(my_image, axis=0)
model.predict(my_image)


