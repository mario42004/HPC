# U-Net Kvasir-SEG Training Guide on MN5

This project trains a U-Net model for binary polyp segmentation with the Kvasir-SEG dataset. The workflow is intentionally split into three clean stages:

1. Dataset splitting with a Slurm batch job.
2. Model training with a separate Slurm batch job.
3. Inference from a Python script or a Jupyter notebook.

The training code is reusable Python code, while the MN5-specific execution details live in the `sbatch/` folder.

## Project Structure

```text
.
├── data/
│   └── kvasir-seg.zip
├── notebooks/
│   └── inference.ipynb
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── splits/
├── sbatch/
│   ├── split_dataset.sbatch
│   └── train.sbatch
├── scripts/
│   ├── inference.py
│   ├── split_dataset.py
│   └── train.py
├── src/
│   └── mn5_segmentation/
│       ├── data.py
│       ├── metrics.py
│       ├── model.py
│       └── utils.py
├── requirements.txt
└── README.md
```

## What Each File Does

- `scripts/split_dataset.py`: scans the dataset and creates reproducible train/validation/test splits.
- `scripts/train.py`: trains the U-Net model using the split file.
- `scripts/inference.py`: loads a trained checkpoint and generates segmentation masks.
- `notebooks/inference.ipynb`: interactive notebook for visual inference.
- `sbatch/split_dataset.sbatch`: Slurm job for dataset splitting.
- `sbatch/train.sbatch`: Slurm job for training.
- `src/mn5_segmentation/`: shared dataset, model, metric, and utility code.

## 1. Connect to MN5

From your local machine, connect to the MN5 login node with SSH:

```bash
ssh <your-bsc-user>@glogin3.bsc.es
```

If your course, account, or allocation uses another login host, replace `glogin3.bsc.es` with the host provided by BSC.

Once connected, move to the project folder:

```bash
cd /path/to/MN5
```

For this workspace, the project path is:

```bash
cd /home/mariojojoaacosta/Documents/BSC/projects/MN5
```

## 2. Check the Anaconda Module

Before launching the scripts, check which Anaconda modules are available on MN5:

```bash
module avail 2>&1 | grep -i anaconda
```

Then load the appropriate module. For example:

```bash
module load anaconda
```

The provided Slurm scripts already try to load Anaconda automatically:

```bash
module load anaconda 2>/dev/null || true
```

This means the jobs will continue even if the exact module name differs. If MN5 shows a versioned module such as `anaconda/2023.07`, update the `module load` line inside the two files under `sbatch/`.

## 3. Python Environment

If you already have a working Python environment, activate it before testing scripts manually:

```bash
source /path/to/your/environment/bin/activate
```

The Slurm scripts also try to activate this optional environment if it exists:

```bash
source "$HOME/BSC_jobs/bin/activate"
```

Install the required Python packages if needed:

```bash
pip install -r requirements.txt
```

The main requirements are PyTorch, torchvision, NumPy, Pillow, Matplotlib, tqdm, and JupyterLab.

Both Slurm scripts also set:

```bash
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
```

This allows Python to import the local `mn5_segmentation` package when the job runs from the project root.

## 4. Prepare the Dataset

Place the Kvasir-SEG zip archive here:

```text
data/kvasir-seg.zip
```

The code can read the dataset directly from the zip file, so extraction is not required.

Alternatively, you may use an extracted dataset folder under `data/`, as long as it contains:

```text
images/
masks/
```

Example accepted layouts:

```text
data/kvasir-seg.zip
```

or:

```text
data/Kvasir-SEG/images/
data/Kvasir-SEG/masks/
```

## 5. Create the Dataset Split

Submit the split job from the project root:

```bash
sbatch sbatch/split_dataset.sbatch
```

This creates:

```text
outputs/splits/kvasir_split.json
```

By default, the split is:

```text
80% train
10% validation
10% test
```

The split is reproducible because it uses a fixed seed, currently `999`.

To monitor the job:

```bash
squeue -u $USER
```

To inspect the split log:

```bash
ls logs/
tail -f logs/split_<jobid>.log
```

Replace `<jobid>` with the Slurm job id printed by `sbatch`.

## 6. Train the Model

After the split file exists, launch training:

```bash
sbatch sbatch/train.sbatch
```

The training job requests:

```text
1 node
1 task
40 CPU cores
2 GPUs
1 hour
```

These settings are defined in:

```text
sbatch/train.sbatch
```

The training script uses:

```text
image size: 256
batch size: 8
epochs: 5
learning rate: 0.001
GPUs: 2 when available
```

To monitor the training job:

```bash
squeue -u $USER
```

To follow the training log:

```bash
tail -f logs/train_<jobid>.log
```

The log prints training and validation metrics for every epoch:

```text
train loss
train Dice
train IoU
validation loss
validation Dice
validation IoU
```

## 7. Training Outputs

The trained model checkpoint is saved here:

```text
outputs/checkpoints/unet_kvasir.pt
```

The training history is saved here:

```text
outputs/checkpoints/history.json
```

The checkpoint contains:

- model weights
- training arguments
- epoch history
- elapsed training time

If you run training multiple times, the default checkpoint path will be overwritten. To keep multiple runs, change the `--checkpoint` argument in `sbatch/train.sbatch`, for example:

```bash
--checkpoint outputs/checkpoints/unet_kvasir_${SLURM_JOB_ID}.pt
```

## 8. Run Scripts Manually for Debugging

For quick debugging on an interactive node or local environment, set `PYTHONPATH` and run the scripts directly.

Create a split:

```bash
PYTHONPATH=src python scripts/split_dataset.py \
  --data-root data \
  --output outputs/splits/kvasir_split.json \
  --seed 999
```

Train:

```bash
PYTHONPATH=src python scripts/train.py \
  --data-root data \
  --split-file outputs/splits/kvasir_split.json \
  --checkpoint outputs/checkpoints/unet_kvasir.pt \
  --history outputs/checkpoints/history.json \
  --image-size 256 \
  --batch-size 8 \
  --epochs 5 \
  --learning-rate 0.001 \
  --num-workers 4 \
  --gpus 2
```

For manual training tests without GPUs, use:

```bash
--gpus 1
```

The script will fall back to CPU if CUDA is not available, but CPU training is only recommended for debugging.

## 9. Inference from Python

After training, run inference on a single image:

```bash
PYTHONPATH=src python scripts/inference.py \
  --checkpoint outputs/checkpoints/unet_kvasir.pt \
  --image path/to/image.jpg \
  --output outputs/predictions/prediction.png
```

The predicted binary mask is saved to:

```text
outputs/predictions/prediction.png
```

You can also import the inference functions:

```python
from scripts.inference import load_model, predict_mask, save_prediction

model, device = load_model("outputs/checkpoints/unet_kvasir.pt")
probability, mask = predict_mask(model, "path/to/image.jpg", device)
save_prediction(mask, "outputs/predictions/prediction.png")
```

## 10. Inference from Jupyter

Open:

```text
notebooks/inference.ipynb
```

The notebook:

1. Adds `src/` and `scripts/` to the Python path.
2. Loads `outputs/checkpoints/unet_kvasir.pt`.
3. Selects an example image.
4. Generates a probability map and binary mask.
5. Saves the prediction under `outputs/predictions/`.

If the dataset is only available as `data/kvasir-seg.zip`, the notebook extracts one sample image into `outputs/predictions/` for convenience.

## 11. Useful Slurm Commands

Show your jobs:

```bash
squeue -u $USER
```

Show details for one job:

```bash
scontrol show job <jobid>
```

Cancel one job:

```bash
scancel <jobid>
```

Cancel all your jobs carefully:

```bash
scancel -u $USER
```

Check GPU usage from inside an allocation:

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

## 12. Troubleshooting

If `sbatch` fails because the Anaconda module is not found, check available modules:

```bash
module avail 2>&1 | grep -i anaconda
```

Then edit the module line in:

```text
sbatch/split_dataset.sbatch
sbatch/train.sbatch
```

If Python cannot import `mn5_segmentation`, make sure `PYTHONPATH` includes `src`:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

If the split script cannot find the dataset, check that the zip exists:

```bash
ls -lh data/kvasir-seg.zip
```

If CUDA runs out of memory, reduce the batch size in `sbatch/train.sbatch`:

```bash
--batch-size 4
```

If the checkpoint is missing during inference, confirm that training finished successfully:

```bash
ls -lh outputs/checkpoints/
tail -n 50 logs/train_<jobid>.log
```

## Recommended Workflow

Use this sequence for a clean run:

```bash
ssh <your-bsc-user>@glogin3.bsc.es
cd /path/to/MN5
module avail 2>&1 | grep -i anaconda
module load anaconda
sbatch sbatch/split_dataset.sbatch
squeue -u $USER
sbatch sbatch/train.sbatch
tail -f logs/train_<jobid>.log
```

Then use `notebooks/inference.ipynb` or `scripts/inference.py` to test the trained model.
