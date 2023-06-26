import os
from typing_extensions import Literal
from typing import List, Any, Dict
import cv2
import json
from dotenv import load_dotenv
import torch
import supervisely as sly

from ultralytics import YOLO
from ultralytics.yolo.utils.torch_utils import select_device
from pathlib import Path

root_source_path = str(Path(__file__).parents[3])
app_source_path = str(Path(__file__).parents[1])
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

model_weights_options = os.environ["modal.state.modelWeightsOptions"]
pretrained_weights = os.environ["modal.state.selectedModel"].lower()
custom_weights = os.environ["modal.state.weightsPath"]

device = "cuda" if torch.cuda.is_available() else "cpu"

pretrained_weights_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/{pretrained_weights}.pt"


class YOLOModel(sly.nn.inference.ObjectDetection):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. Ultralytics) #######
        weights_path = self.download(pretrained_weights_url)
        self.yolo = YOLO(weights_path, task='detect')
        sly.logger.info(f"Using device: {device}")
        self.yolo.overrides = {'device': select_device(device)}
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA
        self.imgsz = 640
        """
        if
        else:
            default_img_size = 640
            sly.logger.warning(
                f"Image size is not found in model checkpoint. Use default: {default_img_size}"
            )
            imgsz = default_img_size
        """
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. Ultralytics)  ########
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return list(self.yolo.names.values())  # e.g. ["cat", "dog", ...]

    def get_info(self):
        info = super().get_info()
        info["model_name"] = "Ultralytics YOLO"
        info["checkpoint_name"] = pretrained_weights
        info["pretrained_on_dataset"] = (
            "COCO train 2017" if model_weights_options == "pretrained" else "custom"
        )
        info["device"] = self.device.type
        info["sliding_window_support"] = self.sliding_window_mode
        info["half"] = str(self.half)
        info["input_size"] = self.imgsz
        return info

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionBBox]:
        confidence_threshold = settings.get("confidence_threshold", 0.5)
        image = cv2.imread(image_path)  # BGR

        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. Ultralytics) #######
        results = self.yolo.predict(image)[0]  # get predictions from Ultralytics model
        pred_classes = results.boxes.cls.int().cpu().numpy().tolist()
        class_names = self.get_classes()
        pred_class_names = [class_names[pred_class] for pred_class in pred_classes]
        pred_scores = results.boxes.conf.cpu().numpy().tolist()
        pred_bboxes = results.boxes.xyxy.cpu().numpy()
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. Ultralytics)  ########

        results = []
        for score, class_name, bbox in zip(pred_scores, pred_class_names, pred_bboxes):
            # filter predictions by confidence
            if score >= confidence_threshold:
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                results.append(sly.nn.PredictionBBox(class_name, bbox, score))
        return results

settings = {"confidence_threshold": 0.7}
m = YOLOModel(custom_inference_settings=settings)
m.load_on_device(device=device)

if sly.is_production():
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging
    image_path = "../../demo_data/image_01.jpg"
    results = m.predict(image_path, settings)
    vis_path = "../../demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
