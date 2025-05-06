import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
import logging
from typing import Dict, List
from datetime import datetime

class SpectrogramTrainer:
    def __init__(self, model_save_path: str = "trained_models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model_save_path = model_save_path
        
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            
        self.setup_logging()

    # In SpectrogramTrainer class
    def setup_prompts(self):
        self.templates = {
            "subjects": [
            "radar RF spectrogram exhibiting baseline noise patterns characteristic of an unoccupied space",
            "radar spectrogram displaying distinct movement pattern of single person",
            "radar spectrogram showing two separate human movement signatures"
            ],
            "movement": [
            "radar spectrogram showing static background without movement",
            "radar RF spectrogram displaying linear Doppler signatures indicative of consistent directional movement",
            "radar spectrogram displaying scattered non-linear movement patterns"
            ]
        }

    def enhanced_augment_image(self, image):
        augmented = []
        transforms_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-20, 20)),
            transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ]
        return  transforms.Compose(transforms_list)(image)
    
    def generate_synthetic_data(self, image_path):
        base_image = Image.open(image_path).convert('RGB')
        synthetic_images = []

        for _ in range(2):
            augmented = self.enhanced_augment_image(base_image)
            synthetic_images.append(augmented)
        
        return synthetic_images
    

    
    





        
        

    def contrastive_loss(self, image_features, text_features, temperature=0.05):
        # Normalize features
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        logits = torch.matmul(image_features, text_features.t()) / temperature
        labels = torch.arange(logits.size(0), device=self.device)

        return (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss() (logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)) / 2
                                                
        
        # Calculate similarity
        #similarity = torch.matmul(image_features, text_features.t()) / temperature
        #labels = torch.arange(similarity.size(0)).to(self.device)
        
        #return nn.CrossEntropyLoss()(similarity, labels)

    def setup_logging(self):
        logging.basicConfig(
            filename=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_dataset(self, spectrogram_dir: str) -> List[Dict]:
        dataset = []
        image_labels = {
            "Testing.jpg": {
                "activity": "no",
                "subjects": "none",
                "diagonal": "no",
                "random": "no"
            },
            "SingleSubjectRandomMovement.jpg": {
                "activity": "yes",
                "subjects": "single",
                "diagonal": "no",
                "random": "yes"
            },
            "SingleSubjectDiagonalMovement.jpg": {
                "activity": "yes",
                "subjects": "single",
                "diagonal": "yes",
                "random": "no"
            },
            "2subjectsRandomMovement.jpg": {
                "activity": "yes",
                "subjects": "two",
                "diagonal": "no",
                "random": "yes"
            },
            "2subjectsDiagonalMovement.jpg": {
                "activity": "yes",
                "subjects": "two",
                "diagonal": "yes",
                "random": "no"
            }
        }
        
        for image_name, labels in image_labels.items():
            image_path = os.path.join(spectrogram_dir, image_name)
            if os.path.exists(image_path):
                dataset.append({
                    'path': image_path,
                    'labels': labels,
                    'text_description': self.generate_text_description(labels)
                })
                logging.info(f"Loaded image: {image_path}")
            else:
                logging.warning(f"Image not found: {image_path}")
                
        return dataset
    
    def generate_text_description(self, labels: Dict) -> str:
        activity = "movement activity" if labels["activity"] == "yes" else "no activity"
        subjects = f"{labels['subjects']} subject" if labels['subjects'] != "none" else "no subjects"
        movement = ""
        if labels["activity"] == "yes":
            if labels["diagonal"] == "yes":
                movement = "with diagonal movement"
            elif labels["random"] == "yes":
                movement = "with random movement"
                
        return f"A spectrogram showing {activity} with {subjects} {movement}".strip()
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        return augmentation(image)
    
    def train_model(self, dataset: List[Dict], num_epochs: int = 10, learning_rate: float = 2e-5):
        kfold = KFold(n_splits=5, shuffle=True)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logging.info(f"Starting fold {fold + 1}")
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                self.model.train()
                train_loss = 0
                
                for idx in train_idx:
                    item = dataset[idx]
                    image = Image.open(item['path']).convert('RGB')
                    
                    if np.random.random() > 0.5:
                        image = self.augment_image(image)
                    
                    inputs = self.processor(
                        images=image,
                        text=item['text_description'],
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get image and text features
                    image_features = self.model.get_image_features(**{k: v for k, v in inputs.items() if k in ['pixel_values']})
                    text_features = self.model.get_text_features(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
                    
                    # Calculate loss
                    loss = self.contrastive_loss(image_features, text_features)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                val_loss = self.validate_fold(dataset, val_idx)
                
                avg_train_loss = train_loss / len(train_idx)
                logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}")
                logging.info(f"Training Loss: {avg_train_loss:.4f}")
                logging.info(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(fold, epoch, val_loss)
                
                scheduler.step()

    def calculate_loss(self, image_features, text_features):
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate logit scale
        logit_scale = self.model.logit_scale.exp()
        
        # Compute similarity matrices
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Create labels (diagonal matrix)
        labels = torch.arange(len(image_features), device=self.device)
        
        # Calculate cross entropy loss in both directions
        loss_image = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_text = nn.CrossEntropyLoss()(logits_per_text, labels)
        
        # Return average loss
        return (loss_image + loss_text) / 2

    def validate_fold(self, dataset: List[Dict], val_idx: np.ndarray) -> float:
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for idx in val_idx:
                item = dataset[idx]
                image = Image.open(item['path']).convert('RGB')
                
                inputs = self.processor(
                    images=image,
                    text=item['text_description'],
                    return_tensors="pt",
                    padding=True
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get features
                image_features = self.model.get_image_features(**{k: v for k, v in inputs.items() if k in ['pixel_values']})
                text_features = self.model.get_text_features(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
                
                # Calculate loss
                loss = self.calculate_loss(image_features, text_features)
                val_loss += loss.item()
        
        return val_loss / len(val_idx)
    
    def save_model(self, fold: int, epoch: int, val_loss: float):
        """Save the model checkpoint"""
        save_path = os.path.join(
            self.model_save_path, 
            f"model_fold{fold}_epoch{epoch}_loss{val_loss:.4f}.pt"
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'fold': fold,
            'epoch': epoch,
            'val_loss': val_loss
        }, save_path)
        
        # Save processor configuration separately if needed
        processor_path = os.path.join(
            self.model_save_path,
            f"processor_fold{fold}_epoch{epoch}.json"
        )
        self.processor.save_pretrained(processor_path)
        
        logging.info(f"Model saved to {save_path}")
        logging.info(f"Processor saved to {processor_path}")

    def load_saved_model(self, model_path: str):
        """Load a saved model checkpoint"""
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get processor path from model path
        processor_path = model_path.replace(".pt", ".json").replace("model_", "processor_")
        if os.path.exists(processor_path):
            self.processor = CLIPProcessor.from_pretrained(processor_path)
        
        logging.info(f"Loaded model from {model_path}")
        if os.path.exists(processor_path):
            logging.info(f"Loaded processor from {processor_path}")

def main():
    trainer = SpectrogramTrainer()
    dataset = trainer.load_dataset("E:\\Analyzer")
    
    if len(dataset) == 0:
        logging.error("No images found in the dataset!")
        return
    
    trainer.train_model(
        dataset=dataset,
        num_epochs=3,
        learning_rate=2e-5
    )

if __name__ == "__main__":
    main()
