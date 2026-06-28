import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import config

def check_leakage():
    print("Checking Data Leakage (Subject-Level Overlap) between Train and Test splits...")
    synth_dir = os.path.join(config.BASE_DIR, "datasets", "synthetic_full")
    if not os.path.exists(synth_dir):
        print(f"Directory not found: {synth_dir}")
        print("Please run data generation and splitting first.")
        return
        
    train_dir = os.path.join(synth_dir, "train")
    test_dir = os.path.join(synth_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Train or Test split not found.")
        return

    def get_base_names(folder):
        base_names = set()
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.png')):
                    try:
                        base = f.split('_', 1)[1].rsplit('_syn', 1)[0]
                    except:
                        base = f
                    base_names.add(base)
        return base_names

    train_bases = get_base_names(train_dir)
    test_bases = get_base_names(test_dir)

    print(f"Unique Base Names in Train: {len(train_bases)}")
    print(f"Unique Base Names in Test: {len(test_bases)}")

    overlap = train_bases.intersection(test_bases)
    if overlap:
        print(f"[FAIL] Data Leakage Detected! {len(overlap)} subjects are in both train and test.")
        print(f"Overlap Examples: {list(overlap)[:5]}")
    else:
        print("[SUCCESS] Zero Overlap. Subject-level splitting is correctly preventing data leakage.")

if __name__ == "__main__":
    check_leakage()
