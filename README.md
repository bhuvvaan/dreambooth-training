
If running on GCP use the following configurations for your VM instance, _Deep Learning VM for PyTorch 2.3 with CUDA 12.1, M125, Debian 11, Python 3.10, with PyTorch 2.3 and fast.ai preinstalled_

Training data for dreambooth needs to be on Huggingface Hub for easy access from any system. The dataset used in the code below is at [https://huggingface.co/datasets/bhuv1-c/valid-warehouses-dataset](https://huggingface.co/datasets/bhuv1-c/valid-warehouses-dataset).

```bash
#!/bin/bash

# Clone the diffusers repository
git clone https://github.com/bhuvvaan/dreambooth-training.git

cd dreambooth-training

# Create a virtual environment
python3 -m venv diffusion-venv

# Activate the virtual environment
source diffusion-venv/bin/activate

# Install the diffusers package
pip install .

# Install accelerate
pip install accelerate

# Configure accelerate
accelerate config

cd examples/dreambooth

# Install requirements for the dreambooth example
pip install -U -r requirements.txt

# Login to Hugging Face
huggingface-cli login --token #your token here

export MODEL_NAME="CompVis/stable-diffusion-v1-4" # Model on the hub
export INSTANCE_HF_REPO="bhuv1-c/fruits-for-intent" # Dataset on the hub
export INSTANCE_DIR="fruits-for-intent" # Local dataset directory
export OUTPUT_DIR="md-intent-prediction-data-2" # Output folder on the hub

# Download dataset from Hugging Face Hub (only images)
python3 <<EOF
import os
from huggingface_hub import snapshot_download

# Download all files
repo_dir = snapshot_download(
    repo_id="bhuv1-c/fruits-for-intent",  # Correct string format
    local_dir="fruits-for-intent",
    repo_type="dataset",
    local_dir_use_symlinks=False  # Prevents extra cache files
)

# Filter only image files
valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
for root, _, files in os.walk(repo_dir):
    for file in files:
        if not any(file.lower().endswith(ext) for ext in valid_exts):
            os.remove(os.path.join(root, file))  # Delete non-image files

# Check if dataset is present before training
image_files = [f for f in os.listdir(repo_dir) if f.lower().endswith(tuple(valid_exts))]
if not image_files:
    print("Error: No images found in dataset. Exiting.")
    exit(1)

print("Dataset downloaded and cleaned successfully!")
EOF

# Ensure dataset exists before proceeding
if [ ! -d "$INSTANCE_DIR" ]; then
  echo "Error: Dataset folder '$INSTANCE_DIR' does not exist!"
  exit 1
fi

# Remove any .cache directories
if [ -d "$INSTANCE_DIR/.cache" ]; then
  rm -rf "$INSTANCE_DIR/.cache"
  echo "Removed .cache directory."
fi

# Run training
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --instance_prompt="a close-up photo of an apple, two lemon, one lime, one onion, three oranges and pear on a rustic wooden table"\
  --class_prompt="fruits on table" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=3e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --push_to_hub

```

Hyperparameters can be changed according to the users convenience. A helpful guide on hyperparameters can be found at [https://huggingface.co/blog/dreambooth](https://huggingface.co/blog/dreambooth).

For original diffusers readme, refer to [https://github.com/huggingface/diffusers](url)

