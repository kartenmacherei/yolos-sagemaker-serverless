"""
A custom inference script to use with YOLOS and SageMaker
"""

import torch
from transformers import AutoFeatureExtractor, YolosForObjectDetection


def model_fn(model_dir):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model = YolosForObjectDetection.from_pretrained(model_dir)
    return model, feature_extractor


def predict_fn(data, model_and_feat_extractor):
    model, feature_extractor = model_and_feat_extractor

    img = data.pop("inputs", data)
    pixel_values = feature_extractor(img, return_tensors="pt").pixel_values

    with torch.no_grad():
        model_output = model(pixel_values)

    probas = model_output.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    target_sizes = torch.tensor(img.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(model_output, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]["boxes"]
    return {
        "probabilities": probas[keep].tolist(),
        "bounding_boxes": bboxes_scaled[keep].tolist(),
    }
