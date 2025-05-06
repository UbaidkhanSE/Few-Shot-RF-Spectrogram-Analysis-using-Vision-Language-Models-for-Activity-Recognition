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

class DataAugmentationPipeline:
    def __init__(self, p_horizontal=0.5, rotation_degrees=20, translate_range=0.15):
        self.transforms = {
            'basic': transforms.Compose([
                transforms.RandomHorizontalFlip(p=p_horizontal),
                transforms.RandomRotation((-rotation_degrees, rotation_degrees)),
                transforms.RandomAffine(0, translate=(translate_range, translate_range))
            ]),
            'advanced': transforms.Compose([
                transforms.RandomHorizontalFlip(p=p_horizontal),
                transforms.RandomRotation((-rotation_degrees, rotation_degrees)),
                transforms.RandomAffine(0, translate=(translate_range, translate_range)),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        }
    
    def apply_augmentation(self, image, augmentation_type='advanced'):
        """Apply specified augmentation pipeline to image"""
        return self.transforms[augmentation_type](image)

    def generate_variations(self, image, num_variations=2):
        """Generate multiple variations of the same image"""
        variations = []
        for _ in range(num_variations):
            variations.append(self.apply_augmentation(image))
        return variations

class CrossValidationHandler:
    def __init__(self, n_splits=5, shuffle=True):
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle)
        self.current_fold = 0
        self.fold_metrics = []

    def setup_fold(self, dataset):
        """Setup data for current fold"""
        splits = list(self.kfold.split(dataset))
        train_idx, val_idx = splits[self.current_fold]
        return train_idx, val_idx

    def track_fold_performance(self, fold_metrics):
        """Track performance metrics for each fold"""
        self.fold_metrics.append(fold_metrics)
        
    def get_fold_summary(self):
        """Get summary statistics across all folds"""
        return {
            'mean_val_loss': np.mean([m['val_loss'] for m in self.fold_metrics]),
            'std_val_loss': np.std([m['val_loss'] for m in self.fold_metrics]),
            'best_fold': np.argmin([m['val_loss'] for m in self.fold_metrics])
        }

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
        self.setup_prompts()

    def setup_prompts(self):
        self.templates = {
            "subjects": [
                "radar spectrogram showing empty scene with no human signatures",
                "radar spectrogram displaying distinct movement pattern of single person",
                "radar spectrogram showing two separate human movement signatures"
            ],
            "movement": [
                "radar spectrogram showing static background without movement",
                "radar spectrogram with clear linear diagonal walking trajectories",
                "radar spectrogram displaying scattered non-linear movement patterns"
            ]
        }

    def contrastive_learning_pipeline(self, image_features, text_features, temperature=0.05):
        """Explicit contrastive learning pipeline with temperature scaling"""
        # Normalize features
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        
        # Calculate similarity matrix with temperature scaling
        similarity_matrix = torch.matmul(image_features, text_features.t()) / temperature
        
        # Calculate bidirectional loss
        labels = torch.arange(similarity_matrix.size(0), device=self.device)
        i2t_loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        t2i_loss = nn.CrossEntropyLoss()(similarity_matrix.t(), labels)
        
        return (i2t_loss + t2i_loss) / 2

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

    def train_model(self, dataset: List[Dict], num_epochs: int = 10, learning_rate: float = 2e-5):
        cv_handler = CrossValidationHandler(n_splits=5)
        augmentation_pipeline = DataAugmentationPipeline()
        
        for fold in range(cv_handler.kfold.n_splits):
            train_idx, val_idx = cv_handler.setup_fold(dataset)
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            
            fold_metrics = {'train_losses': [], 'val_losses': [], 'val_loss': float('inf')}
            
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                
                for idx in train_idx:
                    item = dataset[idx]
                    image = Image.open(item['path']).convert('RGB')
                    
                    # Apply augmentation
                    image = augmentation_pipeline.apply_augmentation(image)
                    
                    # Process inputs
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
                    
                    # Calculate loss using contrastive learning pipeline
                    loss = self.contrastive_learning_pipeline(image_features, text_features)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                val_loss = self.validate_fold(dataset, val_idx)
                
                # Track metrics
                fold_metrics['train_losses'].append(train_loss / len(train_idx))
                fold_metrics['val_losses'].append(val_loss)
                
                if val_loss < fold_metrics['val_loss']:
                    fold_metrics['val_loss'] = val_loss
                    self.save_model(fold, epoch, val_loss)
                
                scheduler.step()
            
            cv_handler.track_fold_performance(fold_metrics)
            
        # Print final cross-validation summary
        summary = cv_handler.get_fold_summary()
        logging.info(f"Cross-validation summary: {summary}")

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
                
                # Calculate loss using contrastive learning pipeline
                loss = self.contrastive_learning_pipeline(image_features, text_features)
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
        
        processor_path = model_path.replace(".pt", ".json").replace("model_", "processor_")
        if os.path.exists(processor_path):
            self.processor = CLIPProcessor.from_pretrained(processor_path)
        
        logging.info(f"Loaded model from {model_path}")
        if os.path.exists(processor_path):
            logging.info(f"Loaded processor from {processor_path}")

def main():
    trainer = SpectrogramTrainer()
    dataset = trainer.load_dataset("E:\\Data")
    
    if len(dataset) == 0:
        logging.error("No images found in the dataset!")
        return
    
    trainer.train_model(
        dataset=dataset,
        num_epochs=10,
        learning_rate=2e-5
    )

if __name__ == "__main__":
    main()