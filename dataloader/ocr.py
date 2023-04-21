from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
import requests
from PIL import Image

processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

# load image from the IIIT-5k dataset
# url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = Image.open("../data/Images/train/VizWiz_train_00000000.jpg").convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values
outputs = model(pixel_values)

generated_text = processor.batch_decode(outputs.logits)['generated_text']
print('detected text: ', generated_text)