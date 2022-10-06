"""
This module's purpose is to send image data to a serverless SageMaker endpoint and
visualize the results.
"""

import boto3
import matplotlib.pyplot as plt
import numpy
import PIL
import sagemaker
from sagemaker.huggingface.model import HuggingFacePredictor
from sagemaker.serializers import DataSerializer
from utils.get_coco_labels import coco_id2label

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        cl = p.argmax()
        text = f"{coco_id2label(cl.item())}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.savefig("example.png")


def get_predictor(sagemaker_session) -> HuggingFacePredictor:
    image_serializer = DataSerializer(content_type="image/x-image")

    hf_predictor = HuggingFacePredictor(
        endpoint_name="yolos-s-object-detection-serverless",
        sagemaker_session=sagemaker_session,
        serializer=image_serializer,
    )

    return hf_predictor


def infer_image(img_path: str, predictor: HuggingFacePredictor):
    res = predictor.predict(data=img_path)

    return res


if __name__ == "__main__":
    boto3_sess = boto3.Session(profile_name=os.environ.get["SAGEMAKER_PROFILE"])
    sagemaker_session = sagemaker.Session(boto_session=boto3_sess)

    image_path = "example_resized.jpg"
    predictor = get_predictor(sagemaker_session)

    res = infer_image(image_path, predictor)

    image = PIL.Image.open(image_path)
    plot_results(image, numpy.asarray(res["probabilities"]), res["bounding_boxes"])
