# VLM_Based_Zeroshot.py

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
import random
from torch.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = self._load_data()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.image_size = (224, 224)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation((-15, 15)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))], p=0.3)
        ])

    def _load_data(self):
        labels = {
            "Testing.jpg": {"subjects": 0, "activity": 0, "movement": 0},
            "SingleSubjectDiagonalMovement.jpg": {"subjects": 1, "activity": 1, "movement": 1},
            "SingleSubjectRandomMovement.jpg": {"subjects": 1, "activity": 1, "movement": 2},
            "2subjectsDiagonalMovement.jpg": {"subjects": 2, "activity": 1, "movement": 1},
            "2subjectsRandomMovement.jpg": {"subjects": 2, "activity": 1, "movement": 2}
        }
        return [(os.path.join(self.data_dir, img), label) for img, label in labels.items()]

    def mixup(self, img1, img2, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        mixed = lam * img1 + (1 - lam) * img2
        return mixed, lam

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        
        if random.random() < 0.5:
            image = self.transform(image)
        
        image = np.array(image) / 255.0
        
        if random.random() < 0.3:
            idx2 = random.randint(0, len(self.samples)-1)
            img2_path, _ = self.samples[idx2]
            image2 = Image.open(img2_path).convert('RGB')
            image2 = image2.resize(self.image_size)
            image2 = np.array(image2) / 255.0
            image, _ = self.mixup(image, image2)
            
        image = torch.tensor(image).permute(2, 0, 1).float()
        labels = {k: torch.tensor(v) for k, v in labels.items()}
        return image, labels

class SpectrogramAnalyzer:
    def __init__(self):
        logger.info("Initializing SpectrogramAnalyzer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(self.device)
        self.scaler = GradScaler("cuda", 
                                init_scale=2**16,
                                growth_factor=2,
                                backoff_factor=0.5,
                                growth_interval=2000,
                                enabled=True)
        self.setup_categories()

    def setup_categories(self):
        self.subject_categories = [
            "high resolution radar spectrogram showing empty background with no human signatures present",
            "detailed radar spectrogram showing clear single person movement pattern with distinct trajectory",
            "complex radar spectrogram displaying two separate human movement signatures with unique patterns"
        ]
        
        self.activity_categories = [
            "radar spectrogram showing complete absence of movement or activity signatures",
            "radar spectrogram with pronounced movement patterns and clear activity signatures"
        ]
        
        self.movement_categories = [
            "radar spectrogram displaying static background without any movement traces",
            "radar spectrogram showing distinct diagonal movement trajectories with clear direction",
            "radar spectrogram with scattered non-linear movement patterns in multiple directions"
        ]

    def preprocess_image(self, image):
        img_array = np.array(image)
        img_array = exposure.equalize_adapthist(img_array)
        return Image.fromarray((img_array * 255).astype(np.uint8))

    def train_model(self, data_dir, epochs=200, batch_size=8):
        dataset = SpectrogramDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        param_groups = [
            {'params': self.model.vision_model.parameters(), 'lr': 1e-5},
            {'params': self.model.text_model.parameters(), 'lr': 2e-5}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = torch.optim.CosineAnnealingWarmRestarts(optimizer, T_0=20)
        criterion = nn.CrossEntropyLoss()

        warmup_steps = 200
        for epoch in range(epochs):
            if epoch < warmup_steps:
                lr_scale = min(1., float(epoch + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * pg['lr']
                    
            total_loss = 0
            self.model.train()
            
            for images, labels in dataloader:
                images = images.to(self.device)
                
                with autocast():
                    for category in ['subjects', 'activity', 'movement']:
                        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False).to(self.device)
                        text_inputs = self.processor(
                            text=getattr(self, f"{category}_categories"), 
                            return_tensors="pt", 
                            padding=True
                        ).to(self.device)
                        
                        image_features = self.model.get_image_features(**inputs)
                        text_features = self.model.get_text_features(**text_inputs)
                        
                        similarity = image_features @ text_features.t()
                        loss = criterion(similarity, labels[category].to(self.device))
                        
                        self.scaler.scale(loss).backward()
                        
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                total_loss += loss.item()
                
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def analyze_spectrogram(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        
        with torch.no_grad(), autocast():
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