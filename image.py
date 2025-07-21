import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import os
from train import ActivityClassifier, CLIPProcessor

# Activity descriptions dictionary
ACTIVITY_DESCRIPTIONS = { 
    'Away': (
        "Represents a subject moving away from the sensor, characterized by a decreasing Doppler shift. "
        "The signal intensity diminishes over time as the subject increases their distance. "
        "This pattern typically reflects a smooth, retreating motion without abrupt changes. "
        "It is commonly observed in walking or running away scenarios."
    ),
    'Bend': (
        "Complex motion signature showing downward torso movement with vertical acceleration components. "
        "This activity often involves changes in phase due to dynamic shifts in body posture. "
        "The signal amplitude may increase momentarily as the subject bends closer to the sensor. "
        "It is typically observed during activities like picking up objects or stretching downward."
    ),
    'Crawl': (
        "Low-amplitude repetitive motion pattern with multi-limb coordination signatures. "
        "This movement generates distinct, periodic signals due to alternating limb motion. "
        "The signal amplitude remains relatively low, reflecting proximity to the ground. "
        "Commonly observed in child-like crawling or military-style low-profile movement."
    ),
    'Kneel': (
        "Transitional movement showing vertical displacement with phase changes. "
        "The subject's motion involves a shift from standing to a lower posture. "
        "Signal intensity often increases briefly during the transition phase. "
        "This activity is typically linked to prayer positions or tasks involving kneeling."
    ),
    'Limp': (
        "Asymmetric gait pattern with irregular time-frequency distributions. "
        "One side of the body generates a stronger signal due to uneven step patterns. "
        "The signal often exhibits variations in intensity and periodicity. "
        "This motion is characteristic of individuals with an injury or mobility issues."
    ),
    'Pick': (
        "Compound motion combining bend and lift elements with phase transitions. "
        "The signal captures a downward motion followed by an upward lift. "
        "Distinct frequency changes occur as the body transitions between positions. "
        "Commonly observed during actions like picking up objects from the ground."
    ),
    'Scissor': (
        "Periodic leg movement with alternating Doppler signatures. "
        "The signal pattern reflects the back-and-forth motion of the legs. "
        "Frequency and intensity variations align with the periodic nature of the activity. "
        "This activity is common in leg exercises or specific athletic drills."
    ),
    'Sit': (
        "Transitional motion showing vertical displacement and deceleration. "
        "The subject transitions from a higher to a lower position, such as sitting on a chair. "
        "Signal intensity may temporarily spike during abrupt deceleration. "
        "Observed during normal daily activities involving sitting down."
    ),
    'Step': (
        "Regular bipedal locomotion with consistent time-frequency characteristics. "
        "The signal exhibits periodic peaks corresponding to each step. "
        "Frequency variations indicate the speed and rhythm of walking or running. "
        "This motion is the most common and easily identifiable locomotion pattern."
    ),
    'Toes': (
        "Fine motor movement with low-amplitude high-frequency components. "
        "The signal reflects small, precise motions concentrated in the toes. "
        "Frequency patterns are distinct but may blend with other activities. "
        "Common in actions like tip-toeing or subtle balance adjustments."
    ),
    'Toward': (
        "Motion pattern showing positive Doppler shift and increasing signal intensity. "
        "The subject moves closer to the sensor, causing a steady increase in amplitude. "
        "This pattern is typical of approaching movements like walking or running forward. "
        "Useful for detecting subjects advancing toward a specific location."
    )
}

def predict(image, model, processor, activity_classes):
   device = next(model.parameters()).device
   inputs = processor(images=image, return_tensors="pt").to(device)
   
   with torch.no_grad():
       outputs = model(inputs.pixel_values)
       probs = F.softmax(outputs, dim=1)
   
   top_prob, top_idx = probs[0].max(dim=0)
   activity = activity_classes[top_idx]
   
   img_array = np.array(image)
   freq_components = np.fft.fft2(img_array)
   spectral_density = np.abs(freq_components)**2
   
   avg_intensity = np.mean(img_array)
   freq_spread = np.std(spectral_density)
   time_res = img_array.shape[1]
   
   if freq_spread > 1000 and time_res > 500:
       radar_type = "77GHz (High frequency, high resolution)"
   elif 500 <= freq_spread <= 1000:
       radar_type = "24GHz (Medium frequency, standard resolution)"
   else:
       radar_type = "Xethru (Ultra-wideband, high penetration)"
   
   analysis = {
       "Signal Characteristics": {
           "Average Intensity": f"{avg_intensity:.2f}",
           "Frequency Spread": f"{freq_spread:.2f}",
           "Time Resolution": f"{time_res} samples"
       },
       "Motion Analysis": {
           "Primary Pattern": ACTIVITY_DESCRIPTIONS[activity],
           "Confidence Score": f"{top_prob:.1%}",
           "Radar Type": radar_type
       }
   }
   
   output_text = [
       f"Activity: {activity}",
       f"Confidence: {top_prob:.1%}",
       f"\nRadar Type: {radar_type}",
       f"\nDetailed Analysis:",
       f"• Signal Characteristics:",
       f"  - Average Intensity: {analysis['Signal Characteristics']['Average Intensity']}",
       f"  - Frequency Spread: {analysis['Signal Characteristics']['Frequency Spread']}",
       f"  - Time Resolution: {analysis['Signal Characteristics']['Time Resolution']}",
       f"\n• Motion Description:",
       f"  {analysis['Motion Analysis']['Primary Pattern']}"
   ]
   
   return "\n".join(output_text)

def create_interface(model, processor, activity_classes):
   with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as interface:
       gr.Markdown(
           """
           <div style='text-align: center; background-color: #1a237e; padding: 20px; border-radius: 10px;'>
               <h1 style='color: white; margin: 0;'>Spectrogram Image Analysis</h1>
               <p style='color: white; margin: 10px 0 0 0;'>Analyzes human activity patterns and radar characteristics across multiple frequency bands</p>
           </div>
           """
       )
       
       with gr.Row():
           with gr.Column(scale=1):
               input_image = gr.Image(label="Input Spectrogram", type="pil")
               submit_btn = gr.Button("Analyze", variant="primary")
           
           with gr.Column(scale=2):
               with gr.Row():
                   with gr.Group():
                       gr.Markdown("### Primary Analysis") 
                       activity_box = gr.Textbox(label="Detected Activity", interactive=False)
                       confidence_box = gr.Textbox(label="Confidence Score", interactive=False)
                       radar_type_box = gr.Textbox(label="Radar Type", interactive=False)
                   
                   with gr.Group():
                       gr.Markdown("### Signal Characteristics")
                       intensity_box = gr.Textbox(label="Average Intensity", interactive=False)
                       spread_box = gr.Textbox(label="Frequency Spread", interactive=False)
                       resolution_box = gr.Textbox(label="Time Resolution", interactive=False)
               
               with gr.Group():
                   gr.Markdown("### Motion Analysis")
                   pattern_box = gr.Textbox(label="Pattern Description", lines=3, interactive=False)

       def process_image(img):
           if img is None:
               return ["No image provided"] * 7
           
           try:
               analysis = predict(img, model, processor, activity_classes)
               activity, confidence, radar_type = "", "", ""
               intensity, spread, resolution, pattern = "", "", "", ""
               
               sections = analysis.split('\n')
               in_motion_section = False
               
               for i, line in enumerate(sections):
                   line = line.strip()
                   if line.startswith("Activity:"): 
                       activity = line.replace("Activity: ", "")
                   elif line.startswith("Confidence:"):
                       confidence = line.replace("Confidence: ", "")
                   elif line.startswith("Radar Type:"):
                       radar_type = line.replace("Radar Type: ", "")
                   elif line.startswith("- Average Intensity:"):
                       intensity = line.replace("- Average Intensity: ", "")
                   elif line.startswith("- Frequency Spread:"):
                       spread = line.replace("- Frequency Spread: ", "")
                   elif line.startswith("- Time Resolution:"):
                       resolution = line.replace("- Time Resolution: ", "")
                   elif "Motion Description:" in line and i + 1 < len(sections):
                       pattern = sections[i + 1].strip()
               
               if not pattern and activity in ACTIVITY_DESCRIPTIONS:
                   pattern = ACTIVITY_DESCRIPTIONS[activity]
               
               return [activity, confidence, radar_type, intensity, spread, 
                       resolution, pattern]
               
           except Exception as e:
               print(f"Error processing image: {e}")
               return ["Error processing image"] * 7

       submit_btn.click(
           fn=process_image,
           inputs=[input_image],
           outputs=[activity_box, confidence_box, radar_type_box, intensity_box,
                   spread_box, resolution_box, pattern_box]
       )

   return interface

def load_model(model_path="model_weights/best_model.pth"):
    checkpoint = torch.load(model_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = ActivityClassifier(len(checkpoint['activity_classes']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, processor, checkpoint['activity_classes']


if __name__ == "__main__":
    model, processor, activity_classes = load_model()
    interface = create_interface(model, processor, activity_classes)
    interface.launch()