#from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from skimage import io
from cv2 import normalize,imshow,NORM_MINMAX,resize,waitKey
#test_datagen = ImageDataGenerator(rescale=1. / 255)
#validation_generator = test_datagen.flow_from_directory(
#   "testdata",
#    target_size=(129, 129),
#    batch_size=30)
classes=sorted(["pizza","bread","rice","brocoli","pasta"])
model=load_model("prototype.h5")
img=io.imread("img.jpg")
normalizedImg=img
normalizedImg = normalize(img,  normalizedImg, 0, 1, NORM_MINMAX)
normalizedImg=resize(normalizedImg,(129,129))
normalizedImg=normalizedImg.reshape(1,129,129,3)
print(classes[model.predict_classes(normalizedImg)[0]])
#print(model.metrics_names)
#print(model.evaluate_generator(validation_generator,steps=100))
