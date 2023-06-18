import os
import cv2
import numpy as np
import utils

def best_t_map_for_LPIPS2(HR_dir_path, working_dir_path, task_name, DB_name, LPIPS_folder_name, output_folder_name, t_num, t_step, ext='png'):

    file_path_list = utils.get_file_path_list(f'{HR_dir_path}/{DB_name}', ext)
    file_name_list = utils.get_file_name_list(f'{HR_dir_path}/{DB_name}', ext)
    image_num = len(file_path_list)
    utils.mkdir(f'{output_folder_name}/{task_name}/{DB_name}')

    Global_lpips_min_value = []
    Global_lpips_min_t_value = []
    img_LPIPS_set = None
    img_SR_set = None
    for i_idx in range(image_num):
        print(i_idx)
        file_name_base = file_name_list[i_idx]
        LPIPS_avg_t = []

        for t_idx in range(t_num):
            t_idx2 = t_step * t_idx
            task_name_t = f'{task_name}_t{t_idx2:03d}'
            file_path = os.path.join(working_dir_path, task_name_t, LPIPS_folder_name, file_name_base)
            file_path2 = os.path.join(working_dir_path, task_name_t, DB_name,file_name_base)

            img_LPIPS = cv2.imread(file_path)
            img_SR = cv2.imread(file_path2)

            LPIPS_avg_t.append(np.mean(img_LPIPS / 255.0))

            if t_idx == 0:
                img_LPIPS_set = np.expand_dims(img_LPIPS[:, :, 0], axis=0)
                img_SR_set = np.expand_dims(img_SR, axis=0)
            else:
                img_LPIPS_set = np.concatenate((img_LPIPS_set, np.expand_dims(img_LPIPS[:, :, 0], axis=0)), axis=0)
                img_SR_set = np.concatenate((img_SR_set, np.expand_dims(img_SR, axis=0)), axis=0)

        Global_lpips_min_value.append(np.min(LPIPS_avg_t))
        Global_lpips_min_t_value.append(np.argmin(LPIPS_avg_t) * t_step)

        best_local_LPIPS_map = np.min(img_LPIPS_set, axis=0)
        best_local_T_map = np.argmin(img_LPIPS_set, axis=0)

        SR_mixed_with_T_map = np.zeros(img_SR_set.shape[1:4])
        for i in range(img_SR_set.shape[1]):
            for j in range(img_SR_set.shape[2]):
                SR_mixed_with_T_map[i, j, :] = img_SR_set[best_local_T_map[i, j], i, j, :]

        SR_mixed_with_T_map_uint8 = np.array(SR_mixed_with_T_map, dtype=np.uint8)
        filename_SR_mixed_with_T_map_uint8 = f'{output_folder_name}/{task_name}/{DB_name}/{file_name_base[:-4]}.png'
        cv2.imwrite(filename_SR_mixed_with_T_map_uint8, SR_mixed_with_T_map_uint8)

        nor = 255.0 / (t_num - 1)
        best_local_T_map_nor = best_local_T_map*nor

        best_local_T_map_uint8 = np.array(best_local_T_map_nor, dtype=np.uint8)  # 형변환 : flat -> UNIT8

        best_local_LPIPS_map_dn = cv2.resize(best_local_LPIPS_map, dsize=[0,0], fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        best_local_T_map_uint8_dn = cv2.resize(best_local_T_map_uint8, dsize=[0, 0], fx=0.25, fy=0.25,
                                         interpolation=cv2.INTER_CUBIC)

        filename_LPIPS_min_dn = f'{output_folder_name}/{task_name}/{DB_name}/{file_name_base[:-4]}_Local_LPIPS_min_dn.png'
        filename_T_best_dn = f'{output_folder_name}/{task_name}/{DB_name}/{file_name_base[:-4]}_T_OOS_Map_dn.png'
        cv2.imwrite(filename_LPIPS_min_dn, best_local_LPIPS_map_dn)
        cv2.imwrite(filename_T_best_dn, best_local_T_map_uint8_dn)


from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument('-gt', type=str, dest='gt_folder', default=None, help='input_folder')
parser.add_argument('-sr', type=str, dest='sr_folder', default=None, help='save_folder')

options=parser.parse_args()

if __name__ == '__main__':

    HR_dir_path = options.gt_folder
    SR_dir_path = options.sr_folder
    # HR_dir_path = 'E:/exp/dataset/SR_testing_datasets_crop_x4'
    # SR_dir_path = 'F:/tempE/exp/FxSR-PD-M1234-v2/results'

    # task_name = 'ESRGAN-SROT-M1234-v2-4x'
    # DB_name = 'DIV2K_train_HR'
    # output_folder_name = 'T_OOS_Map/ESRGAN-SROT-M1234-v2-4x'
    # LPIPS_folder_name = 'DIV2K_train_HR_LPIPS'

    task_name = 'FxSR-PD-M1234-v2-DF2K_DIV2K_train'
    DB_name = 'DIV2K_train'
    output_folder_name = 'T_OOS_Map'
    LPIPS_folder_name = 'LPIPS'

    t_num = 21
    t_step = 5
    ext = 'png'

    best_t_map_for_LPIPS2(HR_dir_path, SR_dir_path, task_name, DB_name, LPIPS_folder_name, output_folder_name, t_num, t_step, ext)
