from ultralytics import YOLO
import pickle
import torch
from torchvision import transforms


class EmoRecNet(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = YOLO('models/face_detector_yolov8n.pt')
        with open('models/emotion_classifier_resnet.pkl', 'rb') as file:
            self.classifier = pickle.load(file)
        self.models_to_device()

    def models_to_device(self):
        self.classifier = self.classifier.to(self.device).eval()

    def detect_face(self, frame):
        width, height = frame.size
        bboxes = []
        faces = self.detector.predict(source=frame, device=self.device, max_det=1)
        for face in faces:
            bbox_pred = face.boxes.xyxy
            if bbox_pred.nelement() != 0:
                xmin, ymin, xmax, ymax = bbox_pred[0].cpu().tolist()
                xmin = int(max(0, xmin - 5))
                ymin = int(max(0, ymin - 5))
                xmax = int(min(width, xmax + 5))
                ymax = int(min(height, ymax + 5))
                # bboxes.append(bbox_pred[0].cpu().tolist())
                bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def classify_emotions(self, frame, bboxes):
        face_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        emotions = []
        for box in bboxes:
            croped_face = frame.crop(box)
            croped_face = face_transform(croped_face)
            croped_face = croped_face.to(self.device)
            croped_face = torch.unsqueeze(croped_face, 0)
            pred_class = int(torch.argmax(self.classifier(croped_face), dim=1))
            emotions.append(pred_class)
        return emotions

    def predict(self, frame):
        labels = self.get_labels()
        bboxes = self.detect_face(frame)
        emotions = self.classify_emotions(frame, bboxes)
        result = []
        for i in range(len(emotions)):
            res_dict = {'bbox': bboxes[i], 'emotion': labels[emotions[i]]}
            result.append(res_dict)
        return result

    def get_labels(self):
        return {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }
