# Few-Shot RF Spectrogram Analysis using Vision-Language Models for Activity Recognition

## Abstract

Radio frequency (RF) spectrogram analysis for indoor human activity recognition has traditionally relied on supervised learning with larger labeled datasets. We present a novel framework that employs vision language models (VLMs), notably contrastive language-image pre-training (CLIP), achieving comparable accuracy from RF spectrograms with a few-shot learning approach. 

Our framework provides domain-specific prompt engineering and an improved contrastive learning approach that effectively bridges the gap between natural language description and RF signal features. The system allows instantaneous spectrogram analysis using a user-friendly interface that enables classification of movement patterns, activity type, identification of subject presence, and precise descriptions.

## Key Features

🎯 **Few-Shot Learning**: Achieves high accuracy with minimal training data  
🔬 **Vision-Language Integration**: Leverages CLIP for RF spectrogram understanding  
📊 **Multi-Task Classification**: Supports multiple recognition tasks simultaneously  
⚡ **Real-Time Processing**: Instantaneous spectrogram analysis  
🎨 **User-Friendly Interface**: Interactive system for easy operation  

## Performance Metrics

Our experimental results demonstrate promising performance across multiple scenarios, using only **5 base training samples** with data augmentation:

| Task | Accuracy |
|------|----------|
| Subject Detection | **85.5%** |
| Movement Pattern Recognition | **82.3%** |
| Activity Type Classification | **84.2%** |

## System Architecture

The framework consists of several key components:

- **RF Signal Processing**: Converts radio frequency data to spectrograms
- **Vision-Language Model (CLIP)**: Processes spectrograms as images
- **Domain-Specific Prompting**: Tailored prompts for RF analysis
- **Few-Shot Learning Pipeline**: Minimal data training approach
- **Interactive Interface**: User-friendly classification system

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CLIP model
- Required dependencies (see requirements.txt)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/UbaidkhanSE/Few-Shot-RF-Spectrogram-Analysis-using-Vision-Language-Models-for-Activity-Recognition.git
cd Few-Shot-RF-Spectrogram-Analysis-using-Vision-Language-Models-for-Activity-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
   - `best_model.pth` - Main classification model
   - `best_few_shot_model.pth` - Few-shot learning model

## Usage

### Basic Classification

```python
python index.py --input_spectrogram path/to/spectrogram.png --task activity_type
```

### Available Tasks

- **Subject Detection**: Identify presence of human subjects
- **Movement Pattern**: Classify movement types and patterns
- **Activity Type**: Recognize specific activities
- **Multi-Task**: Perform all classifications simultaneously

### Interactive Interface

Run the interactive interface for real-time analysis:

```python
python image.py
```

## Dataset Structure

```
11_class_activity_data/
├── walking/
├── sitting/
├── standing/
├── running/
├── jumping/
├── waving/
├── clapping/
├── bending/
├── falling/
├── lying/
└── no_activity/
```

## Model Architecture

The system employs a novel approach combining:

1. **CLIP Vision Encoder**: Processes RF spectrograms as images
2. **Text Encoder**: Handles domain-specific prompts
3. **Contrastive Learning**: Aligns visual and textual representations
4. **Few-Shot Adapter**: Enables learning with minimal data

## Training

### Few-Shot Training

```bash
python train.py --shots 5 --augmentation True --model clip --task multi_class
```

### Data Augmentation

The system includes specialized augmentation techniques for RF spectrograms:
- Frequency shifting
- Time warping
- Noise injection
- Spectrogram rotation

## Results and Evaluation

### Comparison with Traditional Methods

| Method | Subject Detection | Movement Pattern | Activity Type | Training Samples |
|--------|------------------|------------------|---------------|------------------|
| Traditional CNN | 88.2% | 85.1% | 87.3% | 1000+ |
| **Our Approach** | **85.5%** | **82.3%** | **84.2%** | **5** |

### Key Advantages

- **Reduced Data Requirements**: 99.5% reduction in training samples
- **Faster Deployment**: Minimal training time required
- **Domain Adaptability**: Easy adaptation to new environments
- **Interpretability**: Natural language descriptions of classifications

## File Structure

```
├── index.py              # Main classification script
├── image.py              # Image processing and interface
├── training.log          # Training history and metrics
├── requirements.txt      # Python dependencies
├── radar.mp4            # Demo video file
├── best_model.pth       # Pre-trained main model
├── best_few_shot_model.pth  # Pre-trained few-shot model
├── model_weights/       # Additional model checkpoints
├── radar/              # Radar processing utilities
└── 11_class_activity_data/  # Training and test datasets
```

## Applications

- **Smart Home Systems**: Automated activity monitoring
- **Healthcare**: Patient activity tracking
- **Security Systems**: Intrusion and behavior detection
- **Elderly Care**: Fall detection and daily activity monitoring
- **Research**: Human behavior analysis

## Contributing

We welcome contributions! Please feel free to submit pull requests, report issues, or suggest improvements.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{few_shot_rf_2024,
  title={Few-Shot RF Spectrogram Analysis using Vision-Language Models for Activity Recognition},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI CLIP team
