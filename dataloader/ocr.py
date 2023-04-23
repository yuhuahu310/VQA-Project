from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition, TrOCRProcessor, VisionEncoderDecoderModel
# import requests
from PIL import Image
import numpy as np
import os
import cv2
import argparse
import json
from tqdm import tqdm


def crop_patches(image_path, bounding_boxes, y_tolerance_ratio=0.9):
    image = cv2.imread(image_path)

    cropped_patches = []

    # # Sort the bounding boxes by their y-coordinates (top to bottom)
    # bounding_boxes = sorted(bounding_boxes, key=lambda b: (np.mean([pt[1] for pt in b]), np.mean([pt[0] for pt in b])))

    # Calculate the average height of the bounding boxes
    avg_height = np.mean([max(np.linalg.norm(np.array(box[0]) - np.array(box[3])), np.linalg.norm(np.array(box[1]) - np.array(box[2]))) for box in bounding_boxes])
    
    # Calculate the y-tolerance based on the average height
    y_tolerance = avg_height * y_tolerance_ratio

    # Sort the bounding boxes with a y-tolerance
    def sort_key(box):
        y_mean = np.mean([pt[1] for pt in box])
        return (y_mean // y_tolerance, np.mean([pt[0] for pt in box]))

    bounding_boxes = sorted(bounding_boxes, key=sort_key)

    for box in bounding_boxes:
        # Order the vertices in clockwise order
        vertices = np.array(box)
        center = vertices.mean(axis=0)
        angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        box = vertices[np.argsort(angles)]

        # Calculate the dimensions of the bounding box
        width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))

        # Calculate the perspective transform and apply it to the image
        src = np.array([box[0], box[1], box[3]], dtype=np.float32)
        dst = np.array([[0, 0], [width, 0], [0, height]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        cropped = cv2.warpAffine(image, M, (width, height))

        # Convert the cropped patch to a PIL Image and append it to the list
        cropped_patches.append(Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)))

    return cropped_patches


def save_cropped_patches(cropped_patches, dest_folder, filename_prefix):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for i, patch in enumerate(cropped_patches):
        filename = f"{filename_prefix}_{i}.png"
        file_path = os.path.join(dest_folder, filename)
        patch.save(file_path)

class SceneTextRecognizer:
    def __init__(self, model='alibaba-base'):
        self.model_name = model
        if model == 'alibaba-base':
            self.processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
            self.model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')
        elif model == 'ms-base':
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str')
        elif model == 'ms-large':
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-str')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-str')
        else:
            raise ValueError('unsupported model:', model)
        

    def recognize(self, patches):
        # load image from the IIIT-5k dataset
        # url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
        # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        # image = Image.open("../ocr/test_patches/test2.png").convert("RGB")
        if self.model_name.startswith('alibaba'):
            pixel_values = self.processor(images=patches, return_tensors="pt").pixel_values
            outputs = self.model(pixel_values)
            return self.processor.batch_decode(outputs.logits)['generated_text']
        else:
            assert self.model_name.startswith('ms')
            pixel_values = self.processor(images=patches, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)


def example_run():
    image_path = "../ocr/test_images/VizWiz_train_00000002.jpg"
    bounding_boxes = [
        [(634,451),(765,451),(765,497),(634,497)],
        [(556,498),(767,479),(772,531),(561,550)]
    ]

    cropped_patches = crop_patches(image_path, bounding_boxes)
    save_cropped_patches(cropped_patches, "patch_out", "patch")

    ocr = SceneTextRecognizer()
    out = ocr.recognize(cropped_patches)
    print('detected text: ', out)


if __name__ == '__main__':
    # example_run()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bboxes', type=str, help='file path to bboxes json')
    parser.add_argument('--imgdir', type=str, help='dir path to images')
    parser.add_argument('--out', type=str, help='file path to write detected texts to')
    args = parser.parse_args()

    with open(args.bboxes, 'r') as f:
        all_bboxes = json.load(f)
    
    ocr = SceneTextRecognizer(model='alibaba-base')
    all_detected_texts = {}
    for image_name, bboxes in tqdm(list(all_bboxes.items())):
        if len(bboxes) == 0: continue
        cropped_patches = crop_patches(f'{args.imgdir}/{image_name}', bboxes)
        texts = ocr.recognize(cropped_patches)
        print(image_name, texts)
        all_detected_texts[image_name] = texts
    
    with open(args.out, 'w') as f:
        json.dump(all_detected_texts, f, indent=2)