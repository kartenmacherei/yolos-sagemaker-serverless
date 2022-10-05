def get_coco_labels_dict():
    with open(
        "src/object-detection/yolos/utils/coco-labels-2014_2017.txt"
    ) as label_file:
        labels_list = label_file.read().splitlines()
        return dict(enumerate(labels_list))


def coco_id2label(id: int):
    labels_dict = get_coco_labels_dict()
    return labels_dict[id - 1]
