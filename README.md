# MN5 Kvasir-SEG Training and Inference Guide

This repository contains a clean U-Net workflow for binary polyp segmentation with the Kvasir-SEG dataset on MN5/HPC.

The workflow is:

1. Connect to the HPC login node.
2. Clone the public repository into your home directory.
3. Launch the dataset split job.
4. Launch the training job.
5. Launch a Jupyter inference job.
6. Open the inference notebook through an SSH tunnel on port `8888`.

## Project Layout

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
│   ├── inference_jupyter.sbatch
│   ├── split_dataset.sbatch
│   └── train.sbatch
├── scripts/
│   ├── inference.py
│   ├── split_dataset.py
│   └── train.py
└── src/
    └── mn5_segmentation/
```

## 1. Connect to the HPC and Clone the Repository

From your local machine, connect to the login node. This guide uses `alogin4`:

```bash
ssh <your-bsc-user>@alogin4.bsc.es
```

Clone the public repository with HTTPS:

```bash
git clone https://github.com/mario42004/HPC.git
cd HPC
```

All paths in this guide are relative to the cloned `HPC/` directory. The batch scripts also compute the project path dynamically, so the repository can live anywhere under your home directory.

You can submit jobs from the project root:

```bash
sbatch sbatch/split_dataset.sbatch
```

or from inside the `sbatch/` folder:

```bash
cd sbatch
sbatch split_dataset.sbatch
```

Both modes are supported.

## 2. Check the Anaconda Module

Before launching jobs, check which Anaconda modules are available:

```bash
module avail 2>&1 | grep -i anaconda
```

The batch scripts try to load Anaconda automatically:

```bash
module load anaconda 2>/dev/null || true
```

If your HPC shows a versioned module, edit the `module load` line in the files under `sbatch/`.

## 3. Dataset

The dataset should be available as:

```text
data/kvasir-seg.zip
```

The project reads this zip file directly. You do not need to extract it.

The split script creates a JSON file with train/validation/test image IDs. It does not copy or extract images.

## 4. Launch the Dataset Split

From the project root:

```bash
SPLIT_JOBID=$(sbatch --parsable sbatch/split_dataset.sbatch)
echo "Split job id: ${SPLIT_JOBID}"
```

From inside `sbatch/`:

```bash
SPLIT_JOBID=$(sbatch --parsable split_dataset.sbatch)
echo "Split job id: ${SPLIT_JOBID}"
```

Monitor the job:

```bash
squeue -j "${SPLIT_JOBID}"
```

Check the log from the project root:

```bash
tail -f logs/split_${SPLIT_JOBID}.log
```

If you are inside `sbatch/`, use:

```bash
tail -f ../logs/split_${SPLIT_JOBID}.log
```

Expected output:

```text
outputs/splits/kvasir_split.json
```

The split contains:

```text
80% train
10% validation
10% test
```

## 5. Launch Training

After the split job finishes successfully, launch training.

From the project root:

```bash
TRAIN_JOBID=$(sbatch --parsable sbatch/train.sbatch)
echo "Training job id: ${TRAIN_JOBID}"
```

From inside `sbatch/`:

```bash
TRAIN_JOBID=$(sbatch --parsable train.sbatch)
echo "Training job id: ${TRAIN_JOBID}"
```

Monitor the job:

```bash
squeue -j "${TRAIN_JOBID}"
```

Follow the log from the project root:

```bash
tail -f logs/train_${TRAIN_JOBID}.log
```

If you are inside `sbatch/`, use:

```bash
tail -f ../logs/train_${TRAIN_JOBID}.log
```

Training settings are defined in `sbatch/train.sbatch`:

```text
queue: acc_debug
GPUs: 2
CPU cores: 40
epochs: 30
batch size: 8
image size: 256
learning rate: 0.001
```

Training outputs:

```text
outputs/checkpoints/unet_kvasir.pt
outputs/checkpoints/history.json
```

The checkpoint is required by the inference notebook.

## 6. Launch the Jupyter Inference Job

After training finishes and the checkpoint exists, start Jupyter for inference.

From the project root:

```bash
JUPYTER_JOBID=$(sbatch --parsable sbatch/inference_jupyter.sbatch)
echo "Jupyter job id: ${JUPYTER_JOBID}"
```

From inside `sbatch/`:

```bash
JUPYTER_JOBID=$(sbatch --parsable inference_jupyter.sbatch)
echo "Jupyter job id: ${JUPYTER_JOBID}"
```

Follow the Jupyter log from the project root:

```bash
tail -f logs/jupyter_${JUPYTER_JOBID}.log
```

If you are inside `sbatch/`, use:

```bash
tail -f ../logs/jupyter_${JUPYTER_JOBID}.log
```

The log prints:

- the compute node running Jupyter
- the port, by default `8888`
- the SSH tunnel command
- the Jupyter URL with token

## 7. Create the SSH Tunnel for Jupyter

Keep the Jupyter job running. In a new terminal on your local machine, create the tunnel.

The Jupyter log will print a command like this:

```bash
ssh -L 8888:<compute-node>:8888 <your-bsc-user>@alogin4.bsc.es
```

Replace:

- `<compute-node>` with the compute node printed in `logs/jupyter_${JUPYTER_JOBID}.log`
- `<your-bsc-user>` with your BSC username

Keep this tunnel terminal open.

Then open the Jupyter URL printed in the log in your local browser. It will look similar to:

```text
http://127.0.0.1:8888/lab?token=<token>
```

Open:

```text
notebooks/inference.ipynb
```

## 8. Inference Notebook Exercises

The inference notebook asks a few easy questions:

1. Visualize GPU usage during inference with `nvidia-smi`.
2. Check whether PyTorch is using `cuda` or `cpu`.
3. Measure inference time for one image.
4. Change the probability threshold and observe the mask.
5. Propose one extra inference measurement, such as average latency, GPU memory before/after inference, number of foreground pixels, or CPU vs GPU timing.

The notebook saves predictions under:

```text
outputs/predictions/
```

## 9. Useful Slurm Commands

Show a running job:

```bash
squeue -j "${JOBID}"
```

Check a job that already finished or failed:

```bash
sacct -j "${JOBID}" --format=JobID,JobName,State,ExitCode,Elapsed
```

Cancel a job:

```bash
scancel "${JOBID}"
```

For example:

```bash
JOBID="${TRAIN_JOBID}"
sacct -j "${JOBID}" --format=JobID,JobName,State,ExitCode,Elapsed
```

## 10. Recommended Full Run

From your local machine:

```bash
ssh <your-bsc-user>@alogin4.bsc.es
```

On the HPC:

```bash
git clone https://github.com/mario42004/HPC.git
cd HPC
module avail 2>&1 | grep -i anaconda

SPLIT_JOBID=$(sbatch --parsable sbatch/split_dataset.sbatch)
tail -f logs/split_${SPLIT_JOBID}.log

# Wait until the split job finishes successfully.
TRAIN_JOBID=$(sbatch --parsable sbatch/train.sbatch)
tail -f logs/train_${TRAIN_JOBID}.log

# Wait until training finishes and the checkpoint exists.
JUPYTER_JOBID=$(sbatch --parsable sbatch/inference_jupyter.sbatch)
tail -f logs/jupyter_${JUPYTER_JOBID}.log
```

From a second local terminal, create the SSH tunnel using the compute node printed in the Jupyter log:

```bash
ssh -L 8888:<compute-node>:8888 <your-bsc-user>@alogin4.bsc.es
```

Then open the Jupyter URL printed in the log and run:

```text
notebooks/inference.ipynb
```
