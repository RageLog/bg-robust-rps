"""
zip_datasets_for_colab.py
=========================
This script zips the 'datasets/raw' and 'datasets/backgrounds' folders 
into a single zip file.
You need to manually upload this zip to your Google Drive (RPC_Colab) folder.
"""
import os
import zipfile
import shutil

def zip_dir(dir_path, zipf, arc_prefix=""):
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Determine the path inside the zip
            arcname = os.path.join(arc_prefix, os.path.relpath(file_path, dir_path))
            zipf.write(file_path, arcname)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "datasets")
    raw_dir = os.path.join(dataset_dir, "raw")
    bg_dir = os.path.join(dataset_dir, "backgrounds")
    
    zip_path = os.path.join(base_dir, "rpc_base_data.zip")
    
    # Warn if raw or backgrounds do not exist
    if not os.path.exists(raw_dir) or not os.path.exists(bg_dir):
        print(f"ERROR: Both 'raw' and 'backgrounds' folders must exist inside the datasets folder ({dataset_dir}).")
        return

    print(f"📦 Preparing zip file: {zip_path}")
    print("⏳ This process may take a while depending on file size...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add raw folder to zip (will be formatted as datasets/raw/ inside zip)
        print("  - adding raw folder...")
        zip_dir(raw_dir, zipf, arc_prefix="datasets/raw")
        
        # Add backgrounds folder to zip
        print("  - adding backgrounds folder...")
        zip_dir(bg_dir, zipf, arc_prefix="datasets/backgrounds")
        
    print(f"\n✅ PROCESS COMPLETED!")
    print(f"👉 Now upload the '{zip_path}' file to the 'RPC_Colab' folder on Google Drive.")

if __name__ == "__main__":
    main()
