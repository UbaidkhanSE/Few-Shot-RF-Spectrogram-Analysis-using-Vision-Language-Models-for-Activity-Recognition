import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

class RadarSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, radars=['24GHz', '77GHz', 'Xethru']):
        self.root_dir = root_dir
        self.transform = transform
        self.radars = radars
        self.activities = sorted([d for d in os.listdir(os.path.join(root_dir, radars[0])) 
                               if os.path.isdir(os.path.join(root_dir, radars[0], d))])
        
        self.samples = []
        # Build dataset with samples from all radars
        for activity_idx, activity in enumerate(self.activities):
            sample_files = sorted(os.listdir(os.path.join(root_dir, radars[0], activity)))
            for sample in sample_files:
                # Ensure all radars have this sample
                valid = True
                for radar in radars[1:]:
                    if not os.path.exists(os.path.join(root_dir, radar, activity, sample)):
                        valid = False
                        break
                
                if valid:
                    self.samples.append((activity, activity_idx, sample))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        activity, label, sample = self.samples[idx]
        
        # Load spectrograms from all radars
        radar_spectrograms = []
        for radar in self.radars:
            img_path = os.path.join(self.root_dir, radar, activity, sample)
            image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency
            
            if self.transform:
                image = self.transform(image)
            
            radar_spectrograms.append(image)
        
        return radar_spectrograms, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to standard dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloaders
#r"C:\Users\mk305r\Desktop\Multi-StreamRadar\11_class_activity_data"
dataset = RadarSpectrogramDataset(root_dir=r"E:\RadarProject\11_class_activity_data", transform=transform)
print(f"Dataset size: {len(dataset)}")




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RadarEncoder(nn.Module):
    def __init__(self, pretrained=True):  # Make sure this parameter exists
        super(RadarEncoder, self).__init__()
        # Use ResNet18 as backbone, removing the final FC layer
        # Update for newer PyTorch versions:
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        resnet = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
    def forward(self, x):
        features = self.encoder(x)
        return features.view(-1, self.feature_dim)

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=512):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        
        # Attention weights for each radar type
        self.attention_24g = nn.Linear(feature_dim, 1)
        self.attention_77g = nn.Linear(feature_dim, 1)
        self.attention_xethru = nn.Linear(feature_dim, 1)
        
    def forward(self, f24, f77, fx):
        # Calculate attention scores
        a24 = torch.sigmoid(self.attention_24g(f24))
        a77 = torch.sigmoid(self.attention_77g(f77))
        ax = torch.sigmoid(self.attention_xethru(fx))
        
        # Normalize attention weights
        attn_sum = a24 + a77 + ax
        a24 = a24 / attn_sum
        a77 = a77 / attn_sum
        ax = ax / attn_sum
        
        # Weighted fusion
        fused = a24 * f24 + a77 * f77 + ax * fx
        return fused

class ActivityClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=11):
        super(ActivityClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiRadarFusion(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super(MultiRadarFusion, self).__init__()
        self.radar_24g_encoder = RadarEncoder(pretrained=pretrained)
        self.radar_77g_encoder = RadarEncoder(pretrained=pretrained)
        self.xethru_encoder = RadarEncoder(pretrained=pretrained)
        
        feature_dim = self.radar_24g_encoder.feature_dim
        self.attention_fusion = AttentionFusion(feature_dim=feature_dim)
        self.classifier = ActivityClassifier(feature_dim=feature_dim, num_classes=num_classes)
        
    def forward(self, inputs):
        r24, r77, xethru = inputs
        
        # Extract features from each radar
        f24 = self.radar_24g_encoder(r24)
        f77 = self.radar_77g_encoder(r77)
        fx = self.xethru_encoder(xethru)
        
        # Fusion with attention
        fused = self.attention_fusion(f24, f77, fx)
        
        # Classification
        outputs = self.classifier(fused)
        return outputs
    
    def get_attention_weights(self, inputs):
        r24, r77, xethru = inputs
        
        # Extract features
        f24 = self.radar_24g_encoder(r24)
        f77 = self.radar_77g_encoder(r77)
        fx = self.xethru_encoder(xethru)
        
        # Calculate attention weights
        a24 = torch.sigmoid(self.attention_fusion.attention_24g(f24))
        a77 = torch.sigmoid(self.attention_fusion.attention_77g(f77))
        ax = torch.sigmoid(self.attention_fusion.attention_xethru(fx))
        
        attn_sum = a24 + a77 + ax
        a24 = a24 / attn_sum
        a77 = a77 / attn_sum
        ax = ax / attn_sum
        
        return {'24GHz': a24, '77GHz': a77, 'Xethru': ax}
    
def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model([r24, r77, xethru])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                r24, r77, xethru = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                outputs = model([r24, r77, xethru])
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_loss / total
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return history


def evaluate_model(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    attention_weights = {'24GHz': [], '77GHz': [], 'Xethru': []}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            
            # Get model outputs
            outputs = model([r24, r77, xethru])
            _, preds = torch.max(outputs, 1)
            
            # Get attention weights
            batch_weights = model.get_attention_weights([r24, r77, xethru])
            for k, v in batch_weights.items():
                attention_weights[k].append(v.cpu().numpy())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Aggregate attention weights
    for k in attention_weights:
        attention_weights[k] = np.concatenate(attention_weights[k])
    
    # Attention weights per class
    class_attention = {radar: [] for radar in attention_weights}
    for class_idx in range(len(class_names)):
        class_mask = np.array(all_labels) == class_idx
        for radar in attention_weights:
            if np.any(class_mask):
                class_attention[radar].append(np.mean(attention_weights[radar][class_mask]))
            else:
                class_attention[radar].append(0)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'attention_weights': attention_weights,
        'class_attention': class_attention
    }

# Get activity class names
class_names = dataset.activities


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_radar_importance(class_attention, class_names):
    radars = list(class_attention.keys())
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(14, 8))
    for i, radar in enumerate(radars):
        offset = (i - len(radars)/2 + 0.5) * width
        plt.bar(x + offset, class_attention[radar], width, label=radar)
    
    plt.xlabel('Activity Class')
    plt.ylabel('Average Attention Weight')
    plt.title('Radar Importance by Activity Class')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('radar_importance.png')
    plt.show()

def analyze_results(results, class_names):
    # Print overall accuracy
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], class_names)
    
    # Plot radar importance by class
    plot_radar_importance(results['class_attention'], class_names)
    
    # Calculate per-radar importance
    radar_importance = {}
    for radar in results['class_attention']:
        radar_importance[radar] = np.mean(results['class_attention'][radar])
    
    print("\nOverall Radar Importance:")
    for radar, importance in radar_importance.items():
        print(f"{radar}: {importance:.4f}")


def measure_resource_usage(model, input_shape=(1, 3, 224, 224)):
    import time
    import torch
    from thop import profile
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    dummy_inputs = [
        torch.randn(input_shape).to(device),
        torch.randn(input_shape).to(device),
        torch.randn(input_shape).to(device)
    ]
    
    # Measure FLOPs and parameters
    macs, params = profile(model, inputs=[dummy_inputs])
    
    # Measure inference time
    warmup_runs = 10
    timing_runs = 100
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_inputs)
    
    # Timing
    start_time = time.time()
    for _ in range(timing_runs):
        _ = model(dummy_inputs)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / timing_runs
    
    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    _ = model(dummy_inputs)
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    return {
        'params': params,
        'macs': macs,
        'inference_time': avg_inference_time * 1000,  # ms
        'memory_usage': memory_usage  # MB
    }

from sklearn.metrics import accuracy_score
def evaluate_radar_contribution(model, test_loader):
    """Evaluate individual radar contributions by ablation study"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    radar_combinations = [
        {'name': 'All Radars', 'mask': [1, 1, 1]},
        {'name': '24GHz Only', 'mask': [1, 0, 0]},
        {'name': '77GHz Only', 'mask': [0, 1, 0]},
        {'name': 'Xethru Only', 'mask': [0, 0, 1]},
        {'name': '24GHz + 77GHz', 'mask': [1, 1, 0]},
        {'name': '24GHz + Xethru', 'mask': [1, 0, 1]},
        {'name': '77GHz + Xethru', 'mask': [0, 1, 1]}
    ]
    
    results = {}
    
    for combo in radar_combinations:
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                r24, r77, xethru = [x.to(device) for x in inputs]
                
                # Apply masking for ablation study
                f24 = model.radar_24g_encoder(r24) if combo['mask'][0] else torch.zeros_like(model.radar_24g_encoder(r24))
                f77 = model.radar_77g_encoder(r77) if combo['mask'][1] else torch.zeros_like(model.radar_77g_encoder(r77))
                fx = model.xethru_encoder(xethru) if combo['mask'][2] else torch.zeros_like(model.xethru_encoder(xethru))
                
                # Fusion and classification
                fused = model.attention_fusion(f24, f77, fx)
                outputs = model.classifier(fused)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        results[combo['name']] = accuracy
    
    return results


# Per-Activity Radar Analysis: Examine which radar type works best for each activity
def plot_radar_activity_heatmap(class_attention, class_names):
    import matplotlib.pyplot as plt
    import numpy as np
    
    data = np.array([class_attention['24GHz'], 
                    class_attention['77GHz'],
                    class_attention['Xethru']])
    
    plt.figure(figsize=(12, 8))
    plt.imshow(data, aspect='auto', cmap='YlGnBu')
    plt.colorbar(label='Attention Weight')
    
    # Add text annotations
    for i in range(len(data)):
        for j in range(len(data[0])):
            plt.text(j, i, f"{data[i][j]:.3f}", 
                     ha="center", va="center", color="black")
    
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(3), ['24GHz', '77GHz', 'Xethru'])
    plt.title('Radar Importance by Activity')
    plt.tight_layout()
    plt.savefig('radar_activity_heatmap.png')
    plt.show()



#  Computational Efficiency Analysis: Add code to measure inference speed and model complexity:
def measure_efficiency(model, val_loader):
    import time
    import torch
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Memory usage (only if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = 0  # Will be updated later
    else:
        memory_used = "N/A - CPU only"
    
    # Model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
    
    # Get a batch for testing
    try:
        sample_batch = next(iter(val_loader))
        inputs, _ = sample_batch
        inputs = [i.to(device) for i in inputs]
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(inputs)
        
        # Timing inference
        start = time.time()
        with torch.no_grad():
            for _ in range(100):  # Average over 100 runs
                _ = model(inputs)
        avg_time = (time.time() - start) / 100 * 1000  # ms
        
        # Get memory stats if using CUDA
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        
    except StopIteration:
        avg_time = "Error: Empty dataloader"
    
    print(f"Model size: {model_size:.2f}M parameters")
    print(f"Memory usage: {memory_used if isinstance(memory_used, str) else f'{memory_used:.2f} MB'}")
    print(f"Average inference time: {avg_time if isinstance(avg_time, str) else f'{avg_time:.2f} ms'}")
    
    return {
        'model_size_M': model_size,
        'memory_usage_MB': memory_used,
        'inference_time_ms': avg_time
    }


def perform_cross_validation(model_class, dataset, k=5, num_epochs=10):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from sklearn.model_selection import KFold
    
    # Initialize K-Fold
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # For storing results
    fold_results = []
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get indices
    indices = list(range(len(dataset)))
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f"FOLD {fold+1}/{k}")
        print("-" * 30)
        
        # Create data samplers
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Create dataloaders
        train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler, num_workers=0)
        
        # Initialize model
        model = model_class(num_classes=11)
        model.to(device)
        
        # Define loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                # Move to device
                radar_inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(radar_inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move to device
                    radar_inputs = [inp.to(device) for inp in inputs]
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(radar_inputs)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Final validation accuracy for this fold
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                radar_inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
                outputs = model(radar_inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        fold_acc = correct / total if total > 0 else 0
        fold_results.append(fold_acc)
        print(f"Fold {fold+1} accuracy: {fold_acc:.4f}")
    
    # Calculate average results
    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print(f"Cross-validation complete!")
    print(f"Average accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    
    return fold_results, avg_acc, std_acc



#  Ablation Studies: Test performance with different combinations of radars:
def ablation_study(model, val_loader, class_names):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Define radar combinations to test
    combinations = [
        {'name': 'All Radars', 'mask': [1, 1, 1]},
        {'name': '24GHz Only', 'mask': [1, 0, 0]},
        {'name': '77GHz Only', 'mask': [0, 1, 0]},
        {'name': 'Xethru Only', 'mask': [0, 0, 1]},
        {'name': '24GHz + 77GHz', 'mask': [1, 1, 0]},
        {'name': '24GHz + Xethru', 'mask': [1, 0, 1]},
        {'name': '77GHz + Xethru', 'mask': [0, 1, 1]}
    ]
    
    results = {}
    confusion_matrices = {}
    
    for combo in combinations:
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                # Move to device
                radar_inputs = [inp.to(device) for inp in batch_inputs]
                batch_labels = batch_labels.to(device)
                
                # Generate masked features
                for i, use_radar in enumerate(combo['mask']):
                    if not use_radar:
                        radar_inputs[i] = torch.zeros_like(radar_inputs[i])
                
                # Forward pass
                outputs = model(radar_inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        results[combo['name']] = accuracy
        
        # Calculate confusion matrix (optional)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds)
        confusion_matrices[combo['name']] = cm
    
    # Plot results
    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(names, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Radar Combination')
    plt.xticks(rotation=45)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.tight_layout()
    plt.savefig('radar_combinations_accuracy.png')
    plt.show()
    
    return results, confusion_matrices




#  t-SNE Visualization: Visualize feature space to understand separability
def visualize_features(model, val_loader, class_names):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    features = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            
            # Extract features before fusion
            f24 = model.radar_24g_encoder(r24)
            f77 = model.radar_77g_encoder(r77)
            fx = model.xethru_encoder(xethru)
            
            # Get fused features
            fused = model.attention_fusion(f24, f77, fx)
            
            features.append(fused.cpu().numpy())
            labels_list.append(labels.numpy())
    
    if features:
        features = np.vstack(features)
        labels_array = np.concatenate(labels_list)
        
        # Apply t-SNE
        print("Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, verbose=1)
        features_tsne = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 10))
        for i, activity in enumerate(class_names):
            mask = labels_array == i
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=activity)
        
        plt.legend()
        plt.title('t-SNE Visualization of Fused Features')
        plt.savefig('tsne_features.png')
        plt.show()
    else:
        print("No features extracted. Check if the validation loader is empty.")




#   Communication Simulation: Since you mentioned ISAC (Integrated Sensing and Communication), you could simulate communication aspects:
def simulate_communication_constraints(model, val_loader, snr_levels=[30, 20, 10, 5, 0, -5]):
    """Simulate performance under different SNR conditions"""
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for snr in snr_levels:
        # Function to add noise based on SNR
        def add_noise(signal, snr_db):
            signal_power = torch.mean(signal**2)
            # Avoid division by zero
            if signal_power < 1e-10:
                return signal  
            noise_power = signal_power / (10**(snr_db/10))
            noise = torch.randn_like(signal) * torch.sqrt(noise_power)
            return signal + noise
        
        # Evaluate with noise
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move to device
                clean_inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                # Add noise to simulate communication channel
                noisy_inputs = [add_noise(x, snr) for x in clean_inputs]
                
                # Forward pass
                outputs = model(noisy_inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        if total > 0:
            results[snr] = correct / total
    
    # Plot results
    plt.figure(figsize=(10, 6))
    snrs = list(results.keys())
    accuracies = [results[s] for s in snrs]
    
    plt.plot(snrs, accuracies, marker='o', linestyle='-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy')
    plt.title('Performance under Communication Constraints')
    plt.grid(True)
    
    # Add data labels
    for x, y in zip(snrs, accuracies):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    
    plt.savefig('communication_simulation.png')
    plt.show()
    
    return results

def quantize_model(model):
    import torch
    
    # Make a copy of the model for quantization
    quantized_model = copy.deepcopy(model)
    
    # Set model to evaluation mode
    quantized_model.eval()
    
    # Fuse Conv, BN, Relu layers if applicable
    # This step depends on your specific model architecture
    
    # Specify quantization configuration
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    
    return quantized_model

def compare_model_sizes(original_model, quantized_model):
    import os
    import torch
    
    # Save the original model
    torch.save(original_model.state_dict(), "original_model.pth")
    original_size = os.path.getsize("original_model.pth") / (1024 * 1024)
    
    # Save the quantized model
    torch.save(quantized_model.state_dict(), "quantized_model.pth")
    quantized_size = os.path.getsize("quantized_model.pth") / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.2f}%")
    
    return {'original_size_MB': original_size, 
            'quantized_size_MB': quantized_size,
            'reduction_percentage': (1 - quantized_size/original_size)*100}




if __name__ == '__main__':
    import multiprocessing
    mp.freeze_support()  # Add this line
    
    # Initialize model and train
    model = MultiRadarFusion(num_classes=11)
    # Load the best model
    best_model = MultiRadarFusion(num_classes=11)
    best_model.load_state_dict(torch.load('best_model.pth'))
    
    # Reduce num_workers or set to 0 for Windows
    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Create a test loader with the validation set
    test_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)  
    # Evaluate the model
    results = evaluate_model(best_model, test_loader, class_names)
    # Analyze the results
    analyze_results(results, class_names)
    
    history = train_model(model, train_loader, val_loader, num_epochs=50)
    # Measure resource usage
    resource_metrics = measure_resource_usage(best_model)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    


    # Evaluate radar contributions
    radar_contributions = evaluate_radar_contribution(best_model, test_loader)

    # Call with your model and validation loader
    efficiency_metrics = measure_efficiency(best_model, val_loader)

    # Call with your model and validation loader
    ablation_results, ablation_cms = ablation_study(best_model, val_loader, class_names)

    # Call with your model
    communication_results = simulate_communication_constraints(best_model, val_loader)

    
    # Call it with your model and data
    visualize_features(best_model, val_loader, class_names)

    # Call with your model class and dataset
    cv_results, cv_avg, cv_std = perform_cross_validation(MultiRadarFusion, dataset, k=5, num_epochs=10)

    # Run quantization (note: requires PyTorch 1.8+ and will only work on CPU)
    import copy
    quantized_model = quantize_model(best_model.cpu())

    # Compare model sizes  
    size_comparison = compare_model_sizes(best_model.cpu(), quantized_model)


    # Call it with your results
    plot_radar_activity_heatmap(results['class_attention'], class_names)

    print(f"Model Parameters: {resource_metrics['params'] / 1e6:.2f}M")
    print(f"Computational Complexity: {resource_metrics['macs'] / 1e9:.2f}G MACs")
    print(f"Inference Time: {resource_metrics['inference_time']:.2f} ms")
    print(f"GPU Memory Usage: {resource_metrics['memory_usage']:.2f} MB")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    names = list(radar_contributions.keys())
    accuracies = list(radar_contributions.values())
    plt.bar(names, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Radar Combination')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('radar_combinations.png')
    plt.show()