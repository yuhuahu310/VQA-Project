from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition, TrOCRProcessor, VisionEncoderDecoderModel
# import requests
from PIL import Image
import numpy as np
import os
import cv2
import argparse
import json
from tqdm import tqdm
import torch
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


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
            self.model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base').to(device)
        elif model == 'ms-base':
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str').to(device)
        elif model == 'ms-large':
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-str')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-str').to(device)
        else:
            raise ValueError('unsupported model:', model)
        
    def recognize(self, patches, batch_size):
        results = []
        for batch_start in range(0, len(patches), batch_size):
            batch_end = batch_start + batch_size
            batch = patches[batch_start:batch_end]

            if self.model_name.startswith('alibaba'):
                pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values.to(device)
                outputs = self.model(pixel_values)
                batch_results = self.processor.batch_decode(outputs.logits)['generated_text']

                # garbage collect otherwise will OOM (this evil huggingface class has memory leak!)
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
            else:
                assert self.model_name.startswith('ms')
                pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values.to(device)
                generated_ids = self.model.generate(pixel_values)
                batch_results = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            results.extend(batch_results)

        return results


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
    parser.add_argument('--batchsize', type=int, default=16, help='batch size of recognition inference')
    parser.add_argument('--model', type=str, choices=['alibaba-base', 'ms-base', 'ms-large'], default='alibaba-base', help='recognition model to use')
    args = parser.parse_args()

    with open(args.bboxes, 'r') as f:
        all_bboxes = json.load(f)

    
    print(f'using model: {args.model} with batch size {args.batchsize}')
    ocr = SceneTextRecognizer(model=args.model)
    all_detected_texts = {}

    for image_name, bboxes in tqdm(list(all_bboxes.items())):
        if len(bboxes) == 0: continue
        cropped_patches = crop_patches(f'{args.imgdir}/{image_name}', bboxes)
        texts = ocr.recognize(cropped_patches, batch_size=args.batchsize)
        print(image_name, texts)
        all_detected_texts[image_name] = texts

        with open(args.out, 'w') as f:
            json.dump(all_detected_texts, f, indent=2)