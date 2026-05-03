import os
import glob
from collections import defaultdict

# Raw image directory
raw_dir = "datasets/raw"
classes = ['rock', 'paper', 'scissors']

all_image_names = []
user_ids = set()

for c in classes:
    class_dir = os.path.join(raw_dir, c)
    if os.path.exists(class_dir):
        files = glob.glob(os.path.join(class_dir, "*.jpg"))
        for f in files:
            basename = os.path.basename(f)
            all_image_names.append(basename)
            
            # HaGRID format: e.g. "00_user_XXX_8b4..._uuid.jpg" 
            # Or the uuid is directly the filename.
            # If the filename contains a UUID (32-character hex), split and extract it.
            parts = basename.replace('.jpg', '').split('_')
            
            # In HaGRID, the subject ID is usually UUID formatted.
            for part in parts:
                if len(part) == 32 or len(part) == 36: # uuid length
                    user_ids.add(part)
                elif len(part) > 10: # Fallback: long hashes
                    user_ids.add(part)

if not all_image_names:
    print(f"[ERROR] No images found inside {raw_dir}. (The dataset might not be present in this environment)")
    exit(1)

print(f"=====================================")
print(f"[INFO] DATASET STATISTICS            ")
print(f"=====================================")
print(f"Total Image Count: {len(all_image_names)}")

if user_ids:
    print(f"Distinct Subject Count: {len(user_ids)}")
else:
    print(f"[WARN] Could not extract User IDs from filenames.")

print("\nExample file names:")
for i in range(min(5, len(all_image_names))):
    print(f" - {all_image_names[i]}")

# Save report
report_path = "reports/subject_count_verification.txt"
os.makedirs("reports", exist_ok=True)
with open(report_path, "w") as f:
    f.write(f"HaGRID Dataset Subset Statistics\n")
    f.write(f"-----------------------------------------\n")
    f.write(f"Total Images: {len(all_image_names)}\n")
    f.write(f"Unique Subjects: {len(user_ids) if user_ids else 'Unknown (No Metadata Found)'}\n")
    
print(f"\n[SUCCESS] Report successfully saved to: {report_path}")
