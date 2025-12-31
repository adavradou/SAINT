# SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis

![](vis.png)

This is the code for SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis (CVPR2020). 

```
@InProceedings{Peng_SAINT_CVPR2020,
  author = {Peng, Cheng and Lin, Wei-An and Liao, Haofu and Chellappa, Rama and Zhou, S. Kevin},
  title = {SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

[Paper](https://arxiv.org/abs/2001.00704)

The framework of this codebase is derived from [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch).

---

## 🚀 Quick Start Guide

### Step 1: Setup Environment

**Using venv (Recommended):**
```bash
# Create virtual environment
python3 -m venv saint_env

# Activate it
source saint_env/bin/activate  # macOS/Linux
# saint_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Using conda:**
```bash
conda create --name SAINT python=3.9
conda activate SAINT
pip install -r requirements.txt
```

**Manual installation:**
```bash
pip install torch torchvision torchaudio
pip install tqdm imageio medpy scikit-image
```

### Step 2: Prepare Data

Download data from [Medical Decathlon Challenge](http://medicaldecathlon.com/):
- [Colon dataset](https://drive.google.com/file/d/1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y/view)
- [Liver dataset](https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu/view)
- [Hepatic Vessel dataset](https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view)

**Process the raw data:**
```bash
# Example for Colon dataset
python process/raw_data_process.py data/Task10_Colon data/COLON
```

This will:
- Remove volumes with metal artifacts
- Clip intensity to 4000 min-max range
- Generate slices in Sagittal and Coronal views
- Create `TRAIN_SLICES/`, `TRAIN_VOL/`, and `TEST/` directories

**Verify data preparation:**
```bash
# Count training slices
find data/COLON/TRAIN_SLICES/HR/ -name "*.pt" | wc -l

# Count test volumes
find data/COLON/TEST/HR/ -name "*.pt" | wc -l
```

### Step 3: Train or Evaluate

**Training (CPU only - for Mac/systems without CUDA):**
```bash
python3 main.py --save COLON --loss 1*L1 --model META_MULTI \
    --epochs 150 --batch_size 120 \
    --dir_data data/COLON/ \
    --data_train TRAIN_SLICES --data_test TEST \
    --data_range 1-168960/1-24 \
    --n_colors 3 --cpu --n_threads 0 \
    --RDNconfig C --pre_train SAINT_ckpt/model_comb.pt \
    --patch_size 10 --save_results
```
*Note: `--n_threads 0` disables multiprocessing for data loading, which is more stable on Mac.*

**Training (with GPU):**
```bash
python3 main.py --save COLON --loss 1*L1 --model META_MULTI \
    --epochs 150 --batch_size 120 \
    --dir_data data/COLON/ \
    --data_train TRAIN_SLICES --data_test TEST \
    --data_range 1-168960/1-24 \
    --n_colors 3 --n_GPUs 1 \
    --RDNconfig C --pre_train SAINT_ckpt/model_comb.pt \
    --patch_size 10 --save_results
```

**Evaluation only:**
```bash
python3 main.py --scale 4 --save COLON --loss 1*L1 --model META_MULTI \
    --batch_size 1 --dir_data data/COLON/ \
    --data_train TRAIN_SLICES --data_test TEST \
    --data_range 1-168960/1-24 \
    --n_colors 3 --cpu --n_threads 0 \
    --RDNconfig C --pre_train SAINT_ckpt/model_comb.pt \
    --save_results --stage 2 --test_only
```

---

## 📋 Detailed Documentation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- TorchVision 0.15+
- scikit-image
- imageio
- medpy
- tqdm

### Important Notes

**About `--data_range`:**
- Format: `START_TRAIN-END_TRAIN/START_TEST-END_TEST`
- Example: `1-168960/1-24` means use slices 1-168960 for training, volumes 1-24 for testing
- Replace these numbers with your actual data counts

**About GPU/CPU:**
- Use `--cpu` flag for CPU-only training (required on Mac)
- Use `--n_GPUs N` for GPU training (N = number of GPUs)
- Cannot use both flags together

**About Paths:**
- Use relative paths: `--dir_data data/COLON/`
- Or absolute paths: `--dir_data /path/to/data/COLON/`

### Training Parameters

**Marginal Super Resolution (Stage 1):**
```bash
python3 main.py --save COLON --loss 1*L1 --model META_MULTI \
    --epochs 150 --batch_size 120 \
    --dir_data data/COLON/ \
    --data_train TRAIN_SLICES --data_test TEST \
    --data_range 1-X/1-Y \
    --n_colors 3 --cpu \
    --RDNconfig C --pre_train SAINT_ckpt/model_comb.pt \
    --patch_size 10 --save_results
```

Notes:
- Training upsampling rates are fixed at X4 and X6 (in `data/srdata.py`)
- `data_range`: Replace X with # of training slices, Y with # of test volumes
- Patch size × largest upsampling rate should not exceed smallest volume size (usually 60 slices)
- Pre-trained checkpoint is recommended as starting point

**Residual Fusion Network (Stage 2):**
```bash
python3 main.py --scale 4 --save COLON --loss 1*L1 --model META_MULTI \
    --epochs 150 --batch_size 1 \
    --dir_data data/COLON/ \
    --data_train TRAIN_SLICES --data_test TRAIN_VOL \
    --data_range 1-Y/1-X \
    --n_colors 3 --cpu \
    --RDNconfig C --pre_train SAINT_ckpt/model_comb.pt \
    --save_results --stage 1
```

SAINT is trained as a two-stage method:
1. First, inference on training set to obtain data for second stage
2. Axial slices generated from concatenating Sagittal and Coronal SR volumes
3. Freeze main network and unfreeze RFN component in `model/__init__.py`

Alternative to RFN training:
- Use pretrained RFN in checkpoint (robust for X4 and X6)
- Use average of Sagittal and Coronal volume (~0.2 dB worse than RFN)

### Evaluation

Evaluate with pretrained checkpoint:
```bash
python3 main.py --scale 4 --save COLON --loss 1*L1 --model META_MULTI \
    --batch_size 1 --dir_data data/COLON/ \
    --data_train TRAIN_SLICES --data_test TEST \
    --data_range 1-X/1-Y \
    --n_colors 3 --cpu \
    --RDNconfig C --pre_train SAINT_ckpt/model_comb.pt \
    --save_results --stage 2 --test_only
```

- Supports X4 and X6 upsampling rates (change `--scale` parameter)
- Use `--stage 1` for first stage (MSR) only
- Use `--stage 2` for full pipeline (MSR + RFN)

---

## 🔄 PyTorch 2.x Migration

This codebase has been updated from PyTorch 1.0 to PyTorch 2.x. Key changes:

### What Changed
1. **DataLoader**: Rewrote custom `MSDataLoader` using modern PyTorch 2.x APIs
2. **Deprecated APIs Removed**:
   - `torch.autograd.Variable` (no longer needed)
   - `scipy.misc` (deprecated)
   - `skimage.measure.compare_psnr` → `skimage.metrics.peak_signal_noise_ratio`
3. **TorchVision**: Updated `pretrained=True` → `weights=VGG19_Weights.IMAGENET1K_V1`
4. **Model Wrapper Fix**: Updated `get_model()` to handle both CPU and GPU modes correctly
5. **Python Version**: Updated from 3.7 to 3.9+

### Compatibility
- ✅ Existing model checkpoints work with PyTorch 2.x
- ✅ All training/testing scripts unchanged
- ✅ Data formats remain the same
- ✅ Command-line arguments unchanged

### Legacy Setup (PyTorch 1.0 - Not Recommended)
```bash
conda create --name SAINT python=3.7
conda activate SAINT
pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-linux_x86_64.whl
pip install tqdm imageio medpy scikit-image
```

---

## 🐛 Troubleshooting

**CUDA Error on Mac:**
- Add `--cpu` flag (Macs don't have CUDA support)

**"Can't pickle" error with multiprocessing:**
- Add `--n_threads 0` to disable multiprocessing (more stable on Mac/CPU)
- Alternative: Use `--n_threads 1` for single worker process

**"Invalid literal for int() with base 10: 'X'":**
- Replace X and Y in `--data_range` with actual numbers from your dataset

**Empty data directories:**
- Run the data processing script first: `python process/raw_data_process.py ...`

**Import errors:**
- Make sure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

**"AttributeError: 'MSR_RDN' object has no attribute 'module'":**
- This is fixed in the PyTorch 2.x version - make sure you have the latest code

---

## 📂 Project Structure

```
SAINT/
├── data/                     # Data loading and preprocessing
├── loss/                     # Loss functions
├── model/                    # Model architectures
├── process/                  # Data processing scripts
├── SAINT_ckpt/              # Pre-trained checkpoints
├── main.py                   # Main training/testing script
├── trainer.py                # Training logic
├── utility.py                # Utility functions
├── option.py                 # Command-line arguments
└── requirements.txt          # Python dependencies
```

---

## 📝 Citation

If you use this code, please cite:

```
@InProceedings{Peng_SAINT_CVPR2020,
  author = {Peng, Cheng and Lin, Wei-An and Liao, Haofu and Chellappa, Rama and Zhou, S. Kevin},
  title = {SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
