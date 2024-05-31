from colorExtract import extract
from read import read
from upscale import image_upscale

img_name="plague.png" 
pixelCount=400
notaCount=2000

if __name__ == "__main__":
    w,h=extract(img_name,pixelCount,notaCount)
    read(w,h)
    image_upscale()