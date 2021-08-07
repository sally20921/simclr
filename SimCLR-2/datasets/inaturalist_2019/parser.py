import math
import argparse
from pathlib import Path
import json

from tqdm import tqdm


def get_min_max_categories(cat_images):
    min_cat = {
        'id': None,
        'num_images': None,
    }

    max_cat = {
        'id': None,
        'num_images': None,
    }

    for cat_id in cat_images.keys():
        num_images = len(cat_images[cat_id])

        if max_cat['num_images'] is None or num_images > max_cat['num_images']:
            max_cat['id'] = cat_id
            max_cat['num_images'] = num_images

        if min_cat['num_images'] is None or num_images < min_cat['num_images']:
            min_cat['id'] = cat_id
            min_cat['num_images'] = num_images

        print('images in category %i: %i' % (cat_id, num_images))

    print('min category -> id: %i, num_image: %i' % (min_cat['id'], min_cat['num_images']))
    print('max category -> id: %i, num_image: %i' % (max_cat['id'], max_cat['num_images']))

    return min_cat, max_cat


def generate_id_dict(obj_list):
    obj_dict = {}

    # create a dict image id -> image
    for obj in tqdm(obj_list):
        obj_id = obj['id']

        if obj_id not in obj_dict:
            obj_dict[obj_id] = obj
        else:
            print('found not unique object id: %i, object: %s' % (obj_id, obj))

    return obj_dict


def parse_json(file_path):
    with open(file_path) as f:
        json_data = json.load(f)

        images = json_data['images']
        id_images = generate_id_dict(images)

        categories = json_data['categories']
        id_categories = generate_id_dict(categories)

        annotations = json_data['annotations']
        id_annotations = generate_id_dict(annotations)

        return id_images, id_categories, id_annotations


def get_images_by_categorie(id_categories, id_annotations, id_images):
    cat_images = {}

    # create a dict category id -> image
    for ann_id in tqdm(id_annotations):
        ann = id_annotations[ann_id]
        img_id = ann['image_id']
        cat_id = ann['category_id']

        if cat_id not in cat_images:
            cat_images[cat_id] = []

        cat_images[cat_id].append(id_images[img_id])

    print('generated category id -> image dict. num keys: %i' % len(cat_images))

    return cat_images


def reduce_images_by_categories(cat_images, reduction):
    red_cat_images = {}

    for cat_id in tqdm(cat_images):
        images = cat_images[cat_id]

        red_num_images = math.ceil(len(images) * reduction)
        red_images = images[0:red_num_images]
        red_cat_images[cat_id] = red_images

    return red_cat_images


def generate_json_output(cat_images, json_file_path):
    images_output = []

    for (category, images) in cat_images.items():
        for image in images:
            images_output.append({image['file_name']: category})

    with open(json_file_path, "w") as write_file:
        json.dump(images_output, write_file)


def parse_dataset(args):

    if args.reduction > 1.0:
        args.reduction = 1.0

    sets = ['train', 'val']
    year = 2019

    dataset_dir_path = Path(args.dataset_dir_path)

    if not dataset_dir_path.exists():
        raise FileNotFoundError('dataset_dir_path does not exist: ' % dataset_dir_path)

    for set_name in sets:

        json_file_path = dataset_dir_path.joinpath(set_name + str(year) + '.json')

        if not json_file_path.exists():
            raise FileNotFoundError('json_file_path does not exist: ' % json_file_path)

        id_images, id_categories, id_annotations = parse_json(json_file_path)
        print('read -> num of images: %i' % len(id_images))
        print('read -> num of categories: %i' % len(id_categories))
        print('read -> num of annotations: %i' % len(id_annotations))

        cat_images = get_images_by_categorie(id_categories, id_annotations, id_images)
        min_cat_images, max_cat_images = get_min_max_categories(cat_images)

        if args.reduction < 1.0:
            red_cat_images = reduce_images_by_categories(cat_images, args.reduction)
            min_red_cat, max_red_cat = get_min_max_categories(red_cat_images)
            cat_images = red_cat_images

        # create_image_folder(args.path, image_folder_path, red_cat_images)
        output_file_path = dataset_dir_path.joinpath(json_file_path.stem + '_' + str(args.reduction) + json_file_path.suffix)
        generate_json_output(cat_images, output_file_path)


def get_parser():
    parser = argparse.ArgumentParser(description="Parse and reduce iNaturalis dataset")
    parser.add_argument("dataset_dir_path", help="path to the inaturalis dataset directory")
    parser.add_argument("--reduction", type=float, default=1.0,
                        help="reduced dataset size relative to the original inaturalis dataset")
    return parser


if __name__ == '__main__':
    parse_dataset(get_parser().parse_args())
