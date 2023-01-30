# import labelme2coco




from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import load_json, save_json
from tqdm import tqdm
import os
import logging
from glob import glob
from tqdm import tqdm 

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


def list_files_recursively(directory: str, contains: list = [".json"], verbose: str = True):
    """
        Walk given directory recursively and return a list of file path with desired extension

        Arguments
        -------
            directory : str
                "data/coco/"
            contains : list
                A list of strings to check if the target file contains them, example: ["coco.png", ".jpg", "jpeg"]
            verbose : bool
                If true, prints some results
        
        Returns
        -------
            relative_filepath_list : list
                List of file paths relative to given directory
            abs_filepath_list : list
                List of absolute file paths
    """

    # define verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    # walk directories recursively and find json files
    abs_filepath_list = []
    relative_filepath_list = []

    # r=root, d=directories, f=files
    for r, _, f in os.walk(directory):
        for file in f:
            # check if filename contains any of the terms given in contains list
            if any(strtocheck in file.lower() for strtocheck in contains):
                abs_filepath = os.path.join(r, file)
                abs_filepath_list.append(abs_filepath)
                relative_filepath = abs_filepath.split(directory)[-1]
                relative_filepath_list.append(relative_filepath)

    number_of_files = len(relative_filepath_list)
    folder_name = directory.split(os.sep)[-1]

    verboseprint("There are {} listed files in folder {}.".format(number_of_files, folder_name))

    return relative_filepath_list, abs_filepath_list



def get_coco_from_labelme_folder(
    labelme_folder: str, coco_category_list: List = None
) -> Coco:
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        coco_category_list: start from a predefined coco cateory list
    """
    # get json list
    _, abs_json_path_list = list_files_recursively(labelme_folder, contains=[".json"])
    labelme_json_list = abs_json_path_list

    # init coco object
    coco = Coco()

    if coco_category_list is not None:
        coco.add_categories_from_coco_category_list(coco_category_list)

    # parse labelme annotations
    category_ind = 0
    for json_path in tqdm(
        labelme_json_list, "Converting labelme annotations to COCO format"
    ):
        data = load_json(json_path)
        json_dir = os.path.dirname(json_path)
        image_path = os.path.join(json_dir,data['imagePath'])#str(Path(labelme_folder) / data["imagePath"])
        # get image size
        width, height = Image.open(image_path).size
        label  = os.path.basename(str(Path(json_dir).parent))
        # init coco image
        # coco_image = CocoImage(file_name=data["imagePath"], height=height, width=width)
        coco_image = CocoImage(file_name=image_path, height=height, width=width)
        
        # iterate over annotations
        for shape in data["shapes"]:
            # set category name and id
            category_name = label #shape["label"]
            category_id = None
            for (
                coco_category_id,
                coco_category_name,
            ) in coco.category_mapping.items():
                if category_name == coco_category_name:
                    category_id = coco_category_id
                    break
            # add category if not present
            if category_id is None:
                category_id = category_ind
                coco.add_category(CocoCategory(id=category_id, name=category_name))
                category_ind += 1
            # parse bbox/segmentation
            if shape["shape_type"] == "rectangle":
                x1 = shape["points"][0][0]
                y1 = shape["points"][0][1]
                x2 = shape["points"][1][0]
                y2 = shape["points"][1][1]
                coco_annotation = CocoAnnotation(
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    category_id=category_id,
                    category_name=category_name,
                )
            elif shape["shape_type"] == "polygon":
                segmentation = [np.asarray(shape["points"]).flatten().tolist()]
                coco_annotation = CocoAnnotation(
                    segmentation=segmentation,
                    category_id=category_id,
                    category_name=category_name,
                )
            else:
                raise NotImplementedError(
                    f'shape_type={shape["shape_type"]} not supported.'
                )
            coco_image.add_annotation(coco_annotation)
        coco.add_image(coco_image)

    return coco



def convert(
    labelme_folder: str,
    export_dir: str = "runs/labelme2coco/",
    train_split_rate: float = 1,
):
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        export_dir: path for coco jsons to be exported
        train_split_rate: ration fo train split
    """
    coco = get_coco_from_labelme_folder(labelme_folder)
    if train_split_rate < 1:
        result = coco.split_coco_as_train_val(train_split_rate)
        # export train split
        save_path = str(Path(export_dir) / "train.json")
        save_json(result["train_coco"].json, save_path)
        logger.info(f"Training split in COCO format is exported to {save_path}")
        # export val split
        save_path = str(Path(export_dir) / "val.json")
        save_json(result["val_coco"].json, save_path)
        logger.info(f"Validation split in COCO format is exported to {save_path}")
    else:
        save_path = str(Path(export_dir) / "dataset.json")
        save_json(coco.json, save_path)
        logger.info(f"Converted annotations in COCO format is exported to {save_path}")
        


if __name__ =='__main__':
    # set directory that contains labelme annotations and image files
    labelme_folder = "F:\\data/Tire_data/final/labeled_org/"

    # set export dir
    export_dir = "./export"

    # set train split rate
    train_split_rate = 0.80

    # convert labelme annotations to coco
    # labelme2coco.convert(labelme_folder, export_dir, train_split_rate)
    convert(labelme_folder, export_dir, train_split_rate)
    # _, abs = list_files_recursively('F:\\data/Tire_data/final/labeled_org/', [".json"])
    # print(abs)