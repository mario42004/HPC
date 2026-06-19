# U-Net segmentation training on MN5

This training shows how to train a U-Net model from scratch for binary polyp segmentation with the Kvasir-SEG dataset.

The Slurm script reserves one node with two GPUs so the notebook can demonstrate both modes:

- single-GPU training by setting `requested_gpus = 1`
- two-GPU training by setting `requested_gpus = 2`

## Start Jupyter on MN5

```bash
sbatch jupyter_notebook.sbatch
tail -f jupyter_<jobid>.log
```

Open the Jupyter URL printed in the log, then run:

```text
train_unet_kvasir.ipynb
```

The notebook downloads Kvasir-SEG into `data/` if it is not already present. If compute nodes do not have internet access, download `Kvasir-SEG.zip` manually from the official dataset page and place it at:

```text
data/Kvasir-SEG.zip
```

Then rerun the notebook cell that extracts the dataset.
