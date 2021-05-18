import tkinter as tk
from tkinter import filedialog
import os
import datetime
import shutil
def Read_Folder():
    #root = tk.Tk()
    #root.withdraw()
    #FolderPath = filedialog.askdirectory()
    FolderPath = '/data2/mtang/project/OpenPCDet/tools/visual/batch_2_4_times'
    files = os.listdir(FolderPath)
    files.sort(key=lambda fn: os.path.getmtime(FolderPath+'/'+fn))
    for i in range(len(files)-1):
        new = 'loss_' + str(i+1) + '.npy'
        os.rename(files[i], new)
Read_Folder()
def get_datetime(i):
    d = str((datetime.datetime.now() - datetime.timedelta(days=i)).date()).split("-")
    return(timeoffile)
