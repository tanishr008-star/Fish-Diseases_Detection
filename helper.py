import torchvision
from torchvision import datasets

fish_train = datasets.ImageFolder("./fish-diseases/Train")
fish_classes = fish_train.classes
description = [
    "Caused by bacterial infection leading to red streaks and haemorrhages on the body, fins, and tail. Fish show ulcers and fin rot under poor water quality or stress conditions.",

    "Caused by Aeromonas hydrophila, leading to ulcers, hemorrhages, and bloated abdomen. Common in warm, polluted water; can cause high mortality if untreated.",

    "Results from bacterial infection of the gills, causing breathing difficulty and inflamed gills. Fish often gasp at the surface due to reduced oxygen uptake.",

    "Characterized by white, cotton-like fungal growths on the body, fins, or gills. Occurs mostly in injured or stressed fish and in poor water conditions.",

    "No visible signs of disease; bright coloration, normal breathing, and active swimming. Represents the control or non-infected condition in your dataset.",

    "Caused by external or internal parasites leading to white spots, irritation, or weight loss. Infected fish may rub against surfaces and show abnormal behavior.",

    "Caused by a virus that whitens the tail and muscle region, mainly affecting young fish. Leads to weakness, erratic swimming, and sometimes sudden death."
]
data = []
for i, item in enumerate(fish_classes):
    class_item  ={}
    class_item['disease'] = item
    class_item['description'] = description[i]
    data.append(class_item) 


