from colorExtract import extract
from read import read
from upscale import image_upscale
import os

def generate_file_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"{file_path} generate.")
    else:
        print(f"{file_path} already exist.")

def generateFolders():
    generate_file_path("RGB/R")
    generate_file_path("RGB/G")
    generate_file_path("RGB/B")
    generate_file_path("output")
    generate_file_path("upscale")
    generate_file_path("images")

img_name="plague.png" 
pixelCount=400
notaCount=2000

if __name__ == "__main__":
    generateFolders()
    w,h=extract(img_name,pixelCount,notaCount)
    read(w,h)
    image_upscale()