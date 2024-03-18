import os
import csv
import cv2
import numpy as np
import random
import math
from utils import convert_yolo_bb_to_abs, crop_obj_squared, crop_obj_double_bb, generate_rotated_image, \
    display_image_grid, resize_with_padding, cropp_center
from PIL import Image


def read_image(img_file):
    img = cv2.imread(img_file)
    # print(np.shape(img))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.expand_dims(img, axis=2)
    # print(np.shape(img))

    return img


def read_annot(file_path):
    annot_list = []
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            row = [float(x) for x in row]
            annot_list.append((row[0], row[1:]))
    f.close()
    return annot_list


def read_annot_list(path_list):
    img_path_annot_list = []
    for item in path_list:
        (img_path, annot_path) = item
        img_path_annot_list.append((img_path, read_annot(annot_path)))
    return img_path_annot_list


def make_list(dir_path):
    list_annot = []
    list_img = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            if name.endswith(".txt"):
                annot_file_path = os.path.join(root, name)
                img_file_path = annot_file_path.replace("annot_", "").replace(".txt", ".png")
                if os.path.exists(img_file_path) and os.path.exists(annot_file_path):
                    list_img.append(img_file_path)
                    list_annot.append(annot_file_path)

    if len(list_img) != len(list_annot):
        print("Error : Length of annotation and images list must be equal!!!")
        return []
    else:
        final_list = [(img, annot) for img, annot in zip(list_img, list_annot)]
        return final_list


def generate_datalist(input_path="../output_data/Orig_OPBesteck_dataset_rot_est/"):
    output_list = make_list(input_path)
    print(len(output_list))
    output_list = read_annot_list(output_list)
    print(output_list)
    counter = 1
    data_list = []
    for item in output_list:
        img = read_image((item[0]))
        (img_height, img_width, _) = np.shape(img)
        # print(item[1][0][1])
        bb_list = convert_yolo_bb_to_abs([item[1][0][1]], [img_width, img_height])
        # print(bb_list)
        # img = crop_obj_squared(img,bb_list[0],pad_size=10)
        img = crop_obj_double_bb(img, bb_list[0], pad_size=0)
        diag_size = int(math.sqrt(bb_list[0][2] ** 2 + bb_list[0][3] ** 2)) + 20
        if (np.shape(img)[0] == 0 or np.shape(img)[1] == 0):
            print("Error problemistic sample :", item[0])
        data_list.append([img, diag_size])
        # print(np.shape(img))
        # cv2.imwrite("sample_"+str(counter)+".png",img)
        # counter += 1
        # exit(0)
    print(np.shape(data_list))
    return data_list


def generate_dataset(data_list):
    dataset_img = []
    dataset_label = []
    for item in data_list:
        img = np.array(item[0])
        bb_diag_size = item[1]
        # print(np.shape(img))
        # sample_angles = random.sample(range(-10, 10), 5)
        sample_angles = range(0, 357, 1)
        print(len(sample_angles))
        for sample_angle in sample_angles:
            # print(np.shape(img))
            new_img = generate_rotated_image(img,  # generate a valid rotated image based on sample_angle and
                                             sample_angle,  # it resize image to 64*64
                                             # size=(128, 128),
                                             crop_center=False,
                                             crop_largest_rect=False)
            new_img = resize_with_padding(new_img, bb_diag_size, resize=False)
            new_img = cropp_center(new_img, (bb_diag_size, bb_diag_size))
            new_img = resize_with_padding(new_img, desired_size=144, resize=True)
            dataset_img.append(new_img)
            dataset_label.append(sample_angle)

    N = len(dataset_img)
    random.seed(42)
    sampled_image_indecies = random.sample(range(N), 100)
    # sampled_image_indecies = range(0,101)
    sampled_images = [Image.fromarray(dataset_img[i]) for i in sampled_image_indecies]
    sampled_labels = [dataset_label[i] for i in sampled_image_indecies]
    display_image_grid(sampled_images, 10, sampled_labels, string_="1")

    nb_classes = np.unique(dataset_label).__len__()
    data = np.stack(dataset_img, axis=0)  # converting all images to a np array
    X = data.copy()
    # Y = to_categorical(np.array(sampled_labels) + 45, nb_classes)  # creating categorical labels
    X = X / 255  # move image values to [0, 1] period
    X = np.where(X > 0.5, 1.0, 0.0)
    print("########################", np.shape(X))

    # random.seed(42)
    sampled_image_indecies = random.sample(range(N), 100)
    # sampled_image_indecies = range(0,101)
    sampled_images = [Image.fromarray((X[i] * 255).astype(np.uint8)) for i in sampled_image_indecies]
    sampled_labels = [dataset_label[i] for i in sampled_image_indecies]
    display_image_grid(sampled_images, 10, sampled_labels, string_="2")

    print(len(dataset_img))

    return data, dataset_label


if __name__ == '__main__':
    # test_code()
    # test2()
    # test3()
    l = generate_datalist()
    generate_dataset(l)