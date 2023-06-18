import os
import glob

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def get_file_path_list(path, ext='*'):
    file_path_list = glob.glob(f'{path}/*.{ext}')
    for idx in range(len(file_path_list)):
        file_path_list[idx] = file_path_list[idx].replace('\\','/')
    return file_path_list

def get_file_name_list(path, ext='*'):
    file_path_list = get_file_path_list(path, ext)
    file_name_list = []
    for idx in range(len(file_path_list)):
        file_name_list.append(os.path.basename(file_path_list[idx]))
    return file_name_list

def rename_file(src, dst):
    return os.rename(src, dst)


