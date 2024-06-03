import cv2
import numpy as np
from music21 import stream, note
import time
import threading
import multiprocessing
import os

def dimension(image_path,pixelCount):
    image = cv2.imread(image_path)
    w, h, _ = image.shape
    for i in range(1,51):
        int(h/i)*int(w/i)
        if int(h/i)*int(w/i)<pixelCount:
            h=int(h/i)
            w=int(w/i)
            break
        if i==50:
            h=int(h/50)
            w=int(w/50)
    return w,h,i,image

def extract_colors(w,h,i,image):   
    # Resmi RGB formatına dönüştür
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"resim boyutu {h}x{w}. Resim {i} kat küçültüldü. Pixel sayısı: {h*w:,}")
    ort_sure=3.87*1.67
    print(f'2000 nota count için ortalama dosya süresi {round(ort_sure,1)} saniye. {h*w} pixel için ortalama süre: {round(ort_sure*((h*w)/2000),1)} saniye')
    image = cv2.resize(image, (h, w))
    # Resmi düzleştir
    pixels = image.reshape(-1, 3)
    return pixels

def color_to_note(color):
    r, g, b = color
    pitchR=int(int(np.array(r))/3)+21
    pitchG=int(int(np.array(g))/3)+21
    pitchB=int(int(np.array(b))/3)+21
    duration = 0.01
    return note.Note(pitchR, quarterLength=duration),note.Note(pitchG, quarterLength=duration),note.Note(pitchB, quarterLength=duration)

def create_music_from_image(w,h,i,image,notaCount):
    pixels = extract_colors(w,h,i,image)
    
    music_stream_R_list = []
    music_stream_G_list = []
    music_stream_B_list = []
    uzunluk=len(pixels)

    j=notaCount
    for i in range(0,uzunluk,notaCount):
        if i+notaCount>uzunluk:
            j=uzunluk%notaCount
            r=stream.Stream()
            g=stream.Stream()
            b=stream.Stream()
            for pixel_color in pixels[i:i+j]:
                note_obj_R,note_obj_G,note_obj_B = color_to_note(pixel_color)

                r.append(note_obj_R)
                g.append(note_obj_G)
                b.append(note_obj_B)
            music_stream_R_list.append(r)
            music_stream_G_list.append(g)
            music_stream_B_list.append(b)
        else:
            r=stream.Stream()
            g=stream.Stream()
            b=stream.Stream()
            for pixel_color in pixels[i:i+j]:
                note_obj_R,note_obj_G,note_obj_B = color_to_note(pixel_color)

                r.append(note_obj_R)
                g.append(note_obj_G)
                b.append(note_obj_B)
            music_stream_R_list.append(r)
            music_stream_G_list.append(g)
            music_stream_B_list.append(b)
    return music_stream_R_list,music_stream_G_list,music_stream_B_list

def remove_file_path(file_path):
    if os.path.exists(file_path):
        if os.path.isdir(file_path):  # Directory check
            files = os.listdir(file_path)
            if files:
                for file in files:
                    remove_file_path(os.path.join(file_path, file))
        else:
            os.remove(file_path)  # If it is a file, delete it
    else:
        print(f"{file_path} can't find.")


def write_midi(stream, path, filename, index):
    s=time.time()
    stream.write('midi', fp=os.path.join(path, f"{filename}_{index}.mid"))
    e=time.time()
    print("Dosya yazma süresi:",round(e-s,1))
    # print(f"{filename}_{index}.mid dosyası yazıldı.")

def write_music(music_stream, path, filename):
    remove_file_path(path)
    t = time.time()
    processes = []

    for c, stream_file in enumerate(music_stream):
        process = multiprocessing.Process(target=write_midi, args=(stream_file, path, filename, c))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("Toplam süre:", round(time.time() - t, 1))

def main(w,h,i,image,save_path,notaCount):
    music_streams = create_music_from_image(w,h,i,image,notaCount)
    thread_list = []
    for idx, stream_file in enumerate(music_streams):
        if idx == 0:
            col = "R"
        elif idx == 1:
            col = "G"
        else:
            col = "B"

        thread = threading.Thread(target=write_music, args=(stream_file, os.path.join(save_path, col), f"output_{col}"))
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

def extract(img_name,pixelCount,notaCount):
    s=time.time()
    img_folder_path="images"
    image_path = os.path.join(img_folder_path,img_name)# Lütfen kendi resminizi belirtin
    save_path="RGB"
    w,h,i,image=dimension(image_path,pixelCount)
    main(w,h,i,image,save_path,notaCount)
    print("Tüm programın çalışması: ",round(time.time()-s,1)," saniye sürdü")
    return w,h

if __name__ == "__main__":
    img_name="plague.png" 
    pixelCount=300000
    notaCount=2000
    extract(img_name,pixelCount,notaCount)