from train import ResNet18, load_labels_from_xml
import torch
import os
import cv2
import numpy as np
import pandas as pd

PATH = "model.pt"

model = ResNet18().to('cuda')

checkpoint = torch.load(PATH, map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# change folder_path here to the path to a folder containing the videos you will be evaluating my model on.
folder_path = "test_vids"

video_files = [f for f in os.listdir(folder_path) if f.endswith(".mov")]

preds = pd.DataFrame(columns=['Pred', 'Truth'])

for video in video_files:
    video_path = folder_path + "/" + video
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    labels_path = folder_path + "/" + video[:-3] + "xml"
    labels = load_labels_from_xml(labels_path, num_frames=num_frames)

    frame_idx = 0

    temp_preds = pd.DataFrame(columns=['Pred', 'Truth'])
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.tensor(frame, dtype=torch.float32) / 255.0
        frame = frame.unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            probs = model(frame)
            pred = probs.argmax(1).item()

        temp_preds.loc[len(temp_preds)] = [pred, labels[frame_idx]]

        frame_idx += 1

    cap.release()

    preds = pd.concat([preds, temp_preds], ignore_index=True)

accuracy = 100 * ((preds['Pred'] == preds['Truth']).sum() / len(preds))
print(f"Accuracy: {accuracy}%")