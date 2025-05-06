import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import gradio as gr
import numpy as np
import os
import logging
from torch.utils.data import Dataset, DataLoader
from skimage import exposure
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = self._load_data()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Add standard image size
        self.image_size = (224, 224)  # CLIP's standard input size
        
    def _load_data(self):
        labels = {
            "Testing.jpg": {"subjects": 0, "activity": 0, "movement": 0},
            "SingleSubjectDiagonalMovement.jpg": {"subjects": 1, "activity": 1, "movement": 1}, 
            "SingleSubjectRandomMovement.jpg": {"subjects": 1, "activity": 1, "movement": 2},
            "2subjectsDiagonalMovement.jpg": {"subjects": 2, "activity": 1, "movement": 1},
            "2subjectsRandomMovement.jpg": {"subjects": 2, "activity": 1, "movement": 2}
        }
        return [(os.path.join(self.data_dir, img), label) for img, label in labels.items()]
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()
        labels = {k: torch.tensor(v) for k, v in labels.items()}
        return image, labels

class SpectrogramAnalyzer:
    def __init__(self):
        logger.info("Initializing SpectrogramAnalyzer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        
        self.setup_categories()
        self.train_model("spectrogram")

    def setup_categories(self):
        self.subject_categories = [
            "spectrogram showing empty background with no people",
            "spectrogram showing single person movement pattern",
            "spectrogram showing two distinct people movement patterns"
        ]
        
        self.activity_categories = [
            "spectrogram with no activity or movement",
            "spectrogram with clear movement patterns"
        ]
        
        self.movement_categories = [
            "spectrogram showing no movement",
            "spectrogram showing diagonal movement patterns",
            "spectrogram showing random scattered movement"
        ]

    def preprocess_image(self, image):
        img_array = np.array(image)
        img_array = exposure.equalize_adapthist(img_array)
        return Image.fromarray((img_array * 255).astype(np.uint8))

    def train_model(self, data_dir, epochs=50, batch_size=2):
        dataset = SpectrogramDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        category_map = {
            'subjects': 'subject_categories',
            'activity': 'activity_categories',
            'movement': 'movement_categories'
        }

        for epoch in range(epochs):
            total_epoch_loss = 0
            for images, labels in dataloader:
                images = images.to(self.device)
                batch_loss = 0
                
                for category, attr_name in category_map.items():
                    inputs = self.processor(images=images, return_tensors="pt", do_rescale=False).to(self.device)
                    text_inputs = self.processor(text=getattr(self, attr_name), return_tensors="pt", padding=True).to(self.device)
                    
                    image_features = self.model.get_image_features(**inputs)
                    text_features = self.model.get_text_features(**text_inputs)
                    
                    similarity = torch.matmul(image_features, text_features.t())
                    loss = criterion(similarity, labels[category].to(self.device))
                    batch_loss += loss
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                total_epoch_loss += batch_loss.item()
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_epoch_loss/len(dataloader):.4f}")

    def analyze_spectrogram(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # CLIP's standard size
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            
            results = {}
            for category, prompts in {
                'subjects': self.subject_categories,
                'activity': self.activity_categories,
                'movement': self.movement_categories
            }.items():
                text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**text_inputs)
                
                similarity = torch.nn.functional.cosine_similarity(
                    image_features.unsqueeze(1),
                    text_features,
                    dim=-1
                )
                
                pred_idx = similarity.argmax().item()
                confidence = torch.nn.functional.softmax(similarity, dim=-1)[0][pred_idx].item()
                
                results[f"{category}_confidence"] = confidence * 100
                results[category] = prompts[pred_idx]
                
        return results

def create_interface():
    analyzer = SpectrogramAnalyzer()
    
    def process_image(image_path):
        if image_path is None:
            return "Please upload an image"
        
        results = analyzer.analyze_spectrogram(image_path)
        return f"""Analysis Results:
Subjects: {results['subjects']} ({results['subjects_confidence']:.2f}%)
Activity: {results['activity']} ({results['activity_confidence']:.2f}%)
Movement: {results['movement']} ({results['movement_confidence']:.2f}%)"""

    demo = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="filepath"),
        outputs=gr.Textbox(),
        title="Radar Spectrogram Analyzer",
        description="Upload a spectrogram to analyze movement patterns"
    )
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()