import cv2
from cv2 import dnn_superres

def upscale(imageName): 
    #LAPSRN
    sr = dnn_superres.DnnSuperResImpl_create()
    path = 'model/LapSRN_x8.pb'
    sr.readModel(path)
    sr.setModel('lapsrn',8)
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    image = cv2.imread(imageName)
    upscaled = sr.upsample(image)
    cv2.imwrite("output/"+imageName+'upscaled_test_lapsrn.png', upscaled)
    bicubic = cv2.resize(image, (upscaled.shape[1],upscaled.shape[0]),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("output/"+imageName+'bicubic_test_8x.png',bicubic)