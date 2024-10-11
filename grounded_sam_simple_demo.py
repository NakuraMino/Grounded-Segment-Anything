import cv2
import numpy as np
import supervision as sv

import torch
import torchvision
from datetime import datetime
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = "./assets/demo2.jpg"
CLASSES = ["The running dog"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# load image
image = cv2.imread(SOURCE_IMAGE_PATH)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images = np.stack([image, image, image, image, image], axis=0)

# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=images,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

# annotate image with detections
# box_annotator = sv.BoxAnnotator()
# labels = [
#     f"{CLASSES[class_id]} {confidence:0.2f}" 
#     for _, _, confidence, class_id, _, _ 
#     in detections]
# annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# # save the annotated grounding dino image
# cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)


for d in detections:
    # NMS post process
    print(f"Before NMS: {len(d.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(d.xyxy), 
        torch.from_numpy(d.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()
    d.xyxy = d.xyxy[nms_idx]
    d.confidence = d.confidence[nms_idx]
    d.class_id = d.class_id[nms_idx]
    print(f"After NMS: {len(d.xyxy)} boxes")

# Prompting SAM with detected boxes
def batch_segment(sam_predictor: SamPredictor, images: np.ndarray, detections) -> np.ndarray:
    sam_predictor.set_images(images)
    boxes = np.array([d.xyxy for d in detections])
    masks, scores, logits = sam_predictor.predict(
            box=boxes,
            multimask_output=True
        )
    return masks

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# convert detections to masks
# detections[0].mask = segment(
#     sam_predictor=sam_predictor,
#     image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#     xyxy=detections[0].xyxy
# )

masks = batch_segment(
    sam_predictor=sam_predictor,
    images=images,
    detections = detections,
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _, _ 
    in detections]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# save the annotated grounded-sam image
cv2.imwrite("grounded_sam_annotated_image.jpg", annotated_image)
