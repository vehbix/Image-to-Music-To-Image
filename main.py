from colorExtract import extraxt
from read import read
from upscale import upscale

imageName="rose.jpg"
pixelCount=20000
notaCount=2000

extraxt(imageName,pixelCount,notaCount)
read()
upscale(imageName)