#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import os
import shutil
import xml.etree.ElementTree as ET
import tarfile, zipfile
import pickle
import json
import numpy as np
from PIL import Image


def format_IAM_line():
    """
    Format the IAM dataset at line level with the commonly used split (6,482 for train, 976 for validation and 2,915 for test)
    """
    source_folder = "./iam"
    target_folder = "./iam/lines"
    tar_filename = "lines.tgz"
    line_folder_path = os.path.join(target_folder, "lines")

    tar_path = os.path.join(source_folder, tar_filename)
    if not os.path.isfile(tar_path):
        print("error - {} not found".format(tar_path))
        exit(-1)

    os.makedirs(target_folder, exist_ok=True)
    tar = tarfile.open(tar_path)
    tar.extractall(line_folder_path)
    tar.close()

    set_names = ["train", "valid", "test"]
    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }
    charset = set()

    for set_name in set_names:
        id = 0
        current_folder = os.path.join(target_folder, set_name)
        os.makedirs(current_folder, exist_ok=True)
        xml_path = os.path.join(source_folder, "{}.xml".format(set_name))
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_fold_path = os.path.join(line_folder_path, name.split("-")[0], name)
            img_paths = [os.path.join(img_fold_path, p) for p in sorted(os.listdir(img_fold_path))]
            for i, line in enumerate(page[2]):
                label = line.attrib.get("Value")
                img_name = "{}_{}.png".format(set_name, id)
                gt[set_name][img_name] = {
                    "text": label,
                }
                charset = charset.union(set(label))
                new_path = os.path.join(current_folder, img_name)
                os.replace(img_paths[i], new_path)
                id += 1

    shutil.rmtree(line_folder_path)
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)


def format_READ2016_line():
    """
    Format the READ 2016 dataset at line level with the official split (8,349 for training, 1,040 for validation and 1,138 for test)
    """
    source_folder = "./read2016"
    target_folder = "./read2016/lines"
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    tar_filenames = ["Test-ICFHR-2016.tgz", "Train-And-Val-ICFHR-2016.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    os.rename(os.path.join(target_folder, "PublicData", "Training"), os.path.join(target_folder, "train"))
    os.rename(os.path.join(target_folder, "PublicData", "Validation"), os.path.join(target_folder, "valid"))
    os.rename(os.path.join(target_folder, "Test-ICFHR-2016"), os.path.join(target_folder, "test"))
    os.rmdir(os.path.join(target_folder, "PublicData"))
    for set_name in ["train", "valid", ]:
        for filename in os.listdir(os.path.join(target_folder, set_name, "Images")):
            filepath = os.path.join(target_folder, set_name, "Images", filename)
            if os.path.isfile(filepath):
                os.rename(filepath, os.path.join(target_folder, set_name, filename))
        os.rmdir(os.path.join(target_folder, set_name, "Images"))

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    charset = set()
    for set_name in ["train", "valid", "test"]:
        img_fold_path = os.path.join(target_folder, set_name)
        xml_fold_path = os.path.join(target_folder, set_name, "page")
        i = 0
        for xml_file_name in sorted(os.listdir(xml_fold_path)):
            if xml_file_name.split(".")[-1] != "xml":
                continue
            filename = xml_file_name.split(".")[0]
            img_path = os.path.join(img_fold_path, filename+".JPG")
            xml_file_path = os.path.join(xml_fold_path, xml_file_name)
            xml_root = ET.parse(xml_file_path).getroot()
            img = np.array(Image.open(img_path))
            for text_region in xml_root[1][1:]:
                if text_region.tag.split("}")[-1] != "TextRegion":
                    continue
                for balise in text_region:
                    if balise.tag.split("}")[-1] != "TextLine":
                        continue
                    for sub in balise:
                        if sub.tag.split("}")[-1] == "Coords":
                            points = sub.attrib["points"].split(" ")
                            x_points, y_points = list(), list()
                            for p in points:
                                y_points.append(int(p.split(",")[1]))
                                x_points.append(int(p.split(",")[0]))
                        elif sub.tag.split("}")[-1] == "TextEquiv":
                            line_label = sub[0].text
                    if line_label is None:
                        continue
                    top, bottom, left, right = np.min(y_points), np.max(y_points), np.min(x_points), np.max(x_points)
                    new_img_name = "{}_{}.jpeg".format(set_name, i)
                    new_img_path = os.path.join(img_fold_path, new_img_name)
                    curr_img = img[top:bottom + 1, left:right + 1]
                    Image.fromarray(curr_img).save(new_img_path)
                    gt[set_name][new_img_name] = {"text": line_label, }
                    charset = charset.union(line_label)
                    i += 1
                    line_label = None
            os.remove(img_path)
        shutil.rmtree(xml_fold_path)

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)



def pkl2txt(dataset_name):
    for i in ['train', 'valid', 'test']:
        with open((f"./{dataset_name}/lines/labels.pkl"), "rb") as f:
            a = pickle.load(f)
            for k, v in a['ground_truth'][i].items():
                head = k.split('.')[0]
                text = v['text'].replace('¬', '')
                with open(f'./read2016/lines/{head}.txt', 'a') as t: t.write(text)


def move_files_and_delete_folders(parent_folder):
    """
    Move all files from train, valid, and test folders to the parent folder and delete the empty folders.

    Args:
    parent_folder (str): The directory containing the train, valid, and test folders.
    """

    # Define the folders to be moved
    folders = ["train", "valid", "test"]

    for folder in folders:
        folder_path = os.path.join(parent_folder, folder)

        # Check if the folder exists
        if not os.path.isdir(folder_path):
            print(f"{folder} folder does not exist.")
            continue

        # Move files from the subfolder to the parent folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Move the file to the parent folder
                shutil.move(file_path, os.path.join(parent_folder, filename))

        # Remove the empty folder
        os.rmdir(folder_path)
        print(f"Moved all files from {folder} and deleted the folder.")



if __name__ == "__main__":

    format_READ2016_line()
    pkl2txt('read2016')
    move_files_and_delete_folders("./read2016/lines")

    #format_IAM_line()
    #pkl2txt('iam')
    #move_files_and_delete_folders("./iam/lines")

    # format_LAM_line()
