from PIL import Image
import PIL
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
import torch
import json
from io import BytesIO
import numpy as np
import random

def jpegCompression(image):
    qf = random.randrange(10, 100)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

def normalize(input):
  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.244, 0.255])(input)
  return input

preprocess = transforms.Compose([
  transforms.Lambda(jpegCompression),
  transforms.Resize(224),
  transforms.ToTensor(),
  transforms.Lambda(normalize)
])

input = Image.open("examples/cat.jpg")
input_tensor = preprocess(input)[None, :, :, :]

model = resnet50(pretrained=True)
model.eval()

label = model(input_tensor).max(dim=1)[1].item()
target = 21

# targeted attacks
epsilon = 2.0 / 255
delta = torch.zeros_like(input_tensor, requires_grad=True)
opt = optim.SGD([delta], lr=1e-1)

toPil = transforms.ToPILImage()
toTensor = transforms.ToTensor()

ce = nn.CrossEntropyLoss()
for t in range(100):
  adv = input_tensor + delta
  adv = toTensor(jpegCompression(toPil(adv)))
  pred = model(normalize(adv))
  loss1 = -ce(pred, torch.LongTensor([label]))
  loss2 = ce(pred, torch.LongTensor([target]))
  loss = loss1 + loss2
  if t % 10 == 0:
    print(t, loss.item())
  
  opt.zero_grad()
  loss.backward()
  opt.step() # preform optimization step
  delta.data.clamp(-epsilon, epsilon)

ce = nn.CrossEntropyLoss()
l2 = nn.MSELoss()
for t in range(100):
  adv = input_tensor + delta
  pred = model(normalize(adv))
  loss1 = ce(pred, torch.LongTensor([21]))
  loss2 =  (l2(input_tensor - adv)) / (l2(input_tensor))
  loss = loss1 + loss2
  g = adv.grad.clone()
  loss.backward()
  if t % 10 == 0:
    print(t, loss.item())
  adv -= 1e-1 * g
  adv.data.clamp(-epsilon, epsilon)


with open("examples/imagenet_class_index.json") as f:
  imagenet_classes = {int(i):x[1] for i, x in json.load(f).items()}

max_class = pred.max(dim=1)[1].item()
print("Predicted class: ", imagenet_classes[max_class])
print("Predicted probability: ", nn.Softmax(dim=1)(pred)[0, max_class].item())