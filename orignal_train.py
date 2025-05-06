import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import gradio as gr

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.samples = []
        self.labels = []
        
        for sensor in ['24GHz', '77GHz', 'Xethru']:
            sensor_path = os.path.join(root_dir, sensor)
            for activity in os.listdir(sensor_path):
                activity_path = os.path.join(sensor_path, activity)
                for img_name in os.listdir(activity_path):
                    self.samples.append(os.path.join(activity_path, img_name))
                    self.labels.append(activity)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert('RGB')
        text = self.labels[idx]
        
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0]
        }

def train_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = SpectrogramDataset("11_class_activity_data", processor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(10):
        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model, processor

def create_interface(model, processor):
    def predict(image):
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(pixel_values=inputs.pixel_values)
        probs = outputs.logits_per_image.softmax(dim=1)
        return f"Class: {predicted_class}\nConfidence: {confidence:.2f}"

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(),
        outputs=gr.Textbox(),
        title="Spectrogram Classifier"
    )
    return interface

if __name__ == "__main__":
    model, processor = train_model()
    interface = create_interface(model, processor)
    interface.launch()