"""
plot_vector_graphics.py - Vector Graphics Generator (Publication Quality)
=========================================================================
Theses and peer-reviewed journals typically require vector graphics 
(.pdf, .svg, .eps) rather than raster images (.png, .jpg). 
This script reads the saved metrics and generates high-quality 
vector plots from matplotlib/seaborn.
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Adjust line widths and font sizes for academic formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 6),
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight"
})

def plot_confusion_matrix_from_json(json_path, output_dir):
    """Reads the JSON confusion matrix data and plots a vector graphic."""
    if not os.path.exists(json_path):
        return
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'confusion_matrix' not in data:
            return
            
        cm = np.array(data['confusion_matrix'])
        class_names = data.get('class_names', config.CLASS_NAMES)
        model_name = data.get('model_name', 'Model')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
                    
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        
        base_name = os.path.basename(json_path).replace('.json', '')
        # Save as both PDF and SVG
        pdf_path = os.path.join(output_dir, f"{base_name}_vector.pdf")
        svg_path = os.path.join(output_dir, f"{base_name}_vector.svg")
        
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"[SUCCESS] Vector CM Saved: {pdf_path}")
        
    except Exception as e:
        print(f"[ERROR] Plotting CM failed: {e}")

def plot_training_history_from_json(json_path, output_dir):
    """Reads training history (Accuracy/Loss) and plots it as a vector graphic."""
    if not os.path.exists(json_path):
        return
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        history = data.get('history', {})
        if not history or 'accuracy' not in history:
            return
            
        model_name = data.get('model_name', 'Model')
        epochs = range(1, len(history['accuracy']) + 1)
        
        # Accuracy Plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            plt.plot(epochs, history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
            
        plt.title(f'Training and Validation Accuracy: {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        base_name = os.path.basename(json_path).replace('.json', '')
        acc_pdf_path = os.path.join(output_dir, f"{base_name}_acc_vector.pdf")
        acc_svg_path = os.path.join(output_dir, f"{base_name}_acc_vector.svg")
        plt.savefig(acc_pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(acc_svg_path, format='svg', bbox_inches='tight')
        plt.close()
        
        # Loss Plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
            
        plt.title(f'Training and Validation Loss: {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        loss_pdf_path = os.path.join(output_dir, f"{base_name}_loss_vector.pdf")
        loss_svg_path = os.path.join(output_dir, f"{base_name}_loss_vector.svg")
        plt.savefig(loss_pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(loss_svg_path, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"[SUCCESS] Vector History Saved: {acc_pdf_path} & {acc_svg_path}")
        
    except Exception as e:
        print(f"[ERROR] Plotting History failed: {e}")

def main():
    print(f"=========================================")
    print(f"[INFO] VECTOR GRAPHICS GENERATOR FOR THESIS")
    print(f"=========================================\n")
    
    reports_dir = os.path.join(config.BASE_DIR, "reports")
    
    json_files = glob.glob(os.path.join(reports_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"[WARN] No readable JSON reports found in {reports_dir}.")
        print("[INFO] Run this again after training is completed to plot the test data.")
        return
        
    print(f"[INFO] Detected {len(json_files)} reports. Starting to plot...\n")
    
    for j_path in json_files:
        output_dir = os.path.dirname(j_path)
        if 'history' in j_path.lower():
            plot_training_history_from_json(j_path, output_dir)
        elif 'report' in j_path.lower() or 'cm' in j_path.lower() or 'eval' in j_path.lower():
            plot_confusion_matrix_from_json(j_path, output_dir)
            
    print(f"\n[SUCCESS] Plotting completed. Vectors saved alongside JSON files in: {reports_dir}")

if __name__ == "__main__":
    main()
