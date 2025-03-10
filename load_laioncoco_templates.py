import os
import random

# Load templates from txt file
def load_templates_from_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Randomly sample templates based on seed and sample size
def get_random_samples(templates, seed, num_samples):
    random.seed(seed)
    return random.sample(templates, min(num_samples, len(templates)))  # Prevent out-of-bounds error

# Constants
LAION_COCO_DIR = './laion_coco_results/' # Path to laioncoco txt file. Please access the data at: https://drive.google.com/file/d/1Cp9IHjRXa53mvYlhKH7khCHrpVLOaEe3/view?usp=sharing
SOURCE_FILE = os.path.join(LAION_COCO_DIR, 'laion_coco_1M_seed_0.txt')  # Source txt file
SEEDS = range(0, 4)
NUM_SAMPLES = 80

# Load templates from txt file
templates = load_templates_from_txt(SOURCE_FILE)
print(f"Total templates: {len(templates)}")

# Generate and save samples for each seed
for seed in SEEDS:
    sampled_templates = get_random_samples(templates, seed, NUM_SAMPLES)
    output_path = os.path.join(LAION_COCO_DIR, f'laion_coco_samples_size_{NUM_SAMPLES}_seed_{seed}.txt')
    
    with open(output_path, 'w') as f:
        f.write("\n".join(sampled_templates))

    print(f"Saved {len(sampled_templates)} samples to {output_path}")