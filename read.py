import os
from PIL import Image
import numpy as np
from music21 import converter
import time
import multiprocessing

# Open and read text file
with open("image_dimensions.txt", "r") as file:
    # Dosyadan verileri okuma
    lines = file.readlines()
    picture_size=(int(lines[0]),int(lines[1]))

def files_names():
    def extract_number(filename):
        return int(filename.split('_')[-1].split('.')[0])
    # List to hold filenames
    fileNames = []
    # Iterate through folders and files
    for folder in os.listdir("RGB"):
        for file in os.listdir("RGB/"+folder):
            fileNames.append(file)
    # Sort the filenames using the custom function
    fileNames.sort(key=extract_number)
    # Print the sorted filenames
    return fileNames

def extract_piece(file):
    file = os.path.join("RGB", file[7], file)
    piece = converter.parse(file)
    return [piece, file[13]]

def files_to_parse(filenames):
    s = time.time()
    piece_R_lis=[]
    piece_G_lis=[]
    piece_B_lis=[]

    pool = multiprocessing.Pool(processes=3)
    results = pool.map(extract_piece, filenames)
    pool.close()
    pool.join()

    for result in results:
        son,harf=result
        if harf == "R":
            piece_R_lis.append(son)
        elif harf == "G":
            piece_G_lis.append(son)
        elif harf == "B":
            piece_B_lis.append(son)
    
    e = time.time()
    print("Total time to parse files:", round(e - s, 1))
    return piece_R_lis, piece_G_lis, piece_B_lis

def notes(piece_R_lis,piece_G_lis,piece_B_lis):
    notes_R=[]
    notes_G=[]
    notes_B=[]
    for piece in piece_R_lis:
        notes_R.append(piece.flat.notes)

    for piece in piece_G_lis:
        notes_G.append(piece.flat.notes)

    for piece in piece_B_lis:
        notes_B.append(piece.flat.notes)
    return notes_R,notes_G,notes_B

def note_to_number(note):
    notes_per_octave = 12

    pitch_name = note.pitch.pitchClass  # Note name (value between 0-11)
    octave = note.pitch.octave          # octave number
    
    note_number = (octave + 1) * notes_per_octave + pitch_name # Nota numarasını hesapla

    return note_number

def solve(notes):
    pixelList=[]
    text_representation = ''
    count=0
    for note in notes:
        text_representation += str(note) + '\n'

        for not_Alp in note:
            note_number = note_to_number(not_Alp)
            note_number-=21
            note_number*=3
            if note_number>255:
                note_number=255
            color=(note_number)
            pixelList.append(color)
            count+=1

    return pixelList

def multi_solve(notes_R,notes_G,notes_B):
    r_list=[]
    g_list=[]
    b_list=[]
    for notes in notes_R:
        r=solve(notes)
        r_list.append(r)
    for notes in notes_G:
        g=solve(notes)
        g_list.append(g)
    for notes in notes_B:
        b=solve(notes)
        b_list.append(b)
    return r_list,g_list,b_list

def list_merge(liste):
    merged_list = []
    for sub_list in liste:
        merged_list.extend(sub_list)
    return merged_list

def multi_list_merge(r_list,g_list,b_list):
    r_list=list_merge(r_list)
    g_list=list_merge(g_list)
    b_list=list_merge(b_list)
    return r_list,g_list,b_list

def arraying_Color(r,g,b):
    new=[]
    temp=[]
    for i in range(0,len(r)):
        temp.append([r[i],g[i],b[i]])
        if (i+1) % picture_size[0] == 0:
            new.append(temp)
            temp=[]

    array = np.array(new, dtype=np.uint8)
    return array

def image_save(path,new_image):
    os.makedirs(path, exist_ok=True)
    dimensions = f"{str(picture_size[0])}x{str(picture_size[1])}"
    liste=os.listdir(path)
    if len(liste)==0:
        new_file_name=f"output_0_{dimensions}.png"
        save_path=os.path.join(path,new_file_name)
        new_image.save(save_path)
    else:
        liste.sort()
        last_file=liste[-1]
        last_file_num=last_file.split("_")[1]
        new_file_name=f"output_{str(int(last_file_num)+1)}_{dimensions}.png"
        save_path=os.path.join(path,new_file_name)
        new_image.save(save_path)

def main():
    s=time.time()
    print(picture_size)
    filenames=files_names()
    piece_R_lis,piece_G_lis,piece_B_lis=files_to_parse(filenames)
    notes_R,notes_G,notes_B=notes(piece_R_lis,piece_G_lis,piece_B_lis)
    r_list,g_list,b_list=multi_solve(notes_R,notes_G,notes_B)
    r_listMulti,g_listMulti,b_listMulti=multi_list_merge(r_list,g_list,b_list)
    array_RGB=arraying_Color(r_listMulti,g_listMulti,b_listMulti)
    new_image = Image.fromarray(array_RGB)
    image_save("output",new_image)
    e=time.time()
    print("Program running time:",round(e-s,1))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


