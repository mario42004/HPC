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

The notebook is offline-only. It uses `data/kvasir-seg.zip` if it is present, extracts it into `data/`, and also accepts an already extracted dataset under `data/` with `images/` and `masks/` folders. Place the dataset zip at:

```text
data/kvasir-seg.zip
```

Then rerun the notebook cell that extracts the dataset.

## MN5 Jupyter guide

### 1. Submit the job

From the project folder on MN5, submit the Slurm script:

```bash
sbatch jupyter_notebook.sbatch
```

Slurm prints a job id, for example:

```text
Submitted batch job 12345678
```

Use that number in the commands below as `<jobid>`.

### 2. Monitor the queue

Check whether the job is pending or running:

```bash
squeue -u $USER
squeue -j <jobid> -o "%.18i %.9P %.20T %.20R"
```

Useful states:

- `PD`: pending, the job is waiting for resources.
- `R`: running, the node has been assigned.
- `CG`: completing, the job is finishing.

To see the full Slurm details:

```bash
scontrol show job <jobid>
```

### 3. Read the Jupyter log and token

When the job starts, follow the log:

```bash
tail -f jupyter_<jobid>.log
```

The log contains the compute node, the visible GPUs, and the Jupyter URL. Look for a line similar to:

```text
http://127.0.0.1:8888/lab?token=<token>
```

You can also extract only the token/URL lines with:

```bash
grep -E "token=|http://|https://" jupyter_<jobid>.log
```

Copy the full URL or copy the value after `token=`.

### 4. Trainee step: create the SSH tunnel

Each trainee must run this command from a terminal on their local machine, not from inside the MN5 Jupyter job.

Replace `<user>` with your BSC username, `<node>` with the node printed in `jupyter_<jobid>.log`, and keep the port as `8888` unless you changed `PORT` in the sbatch file.

```bash
ssh -L 8888:<node>:8888 <user>@glogin3.bsc.es
```

If your MN5 access requires a different login host, use that host instead of `glogin3.bsc.es`.

Keep this SSH tunnel terminal open while using Jupyter. Then open this address in your local browser:

```text
http://127.0.0.1:8888/lab?token=<token>
```

### 5. Run the training notebook

Open:

```text
train_unet_kvasir.ipynb
```

For the first exercise, keep:

```python
requested_gpus = 1
```

For the second exercise, restart the kernel, change it to:

```python
requested_gpus = 2
```

Then rerun the training cells and compare the total training time and validation Dice score.

### 6. Monitor the running process

To watch the Jupyter output:

```bash
tail -f jupyter_<jobid>.log
```

To open a shell inside the running allocation:

```bash
srun --jobid=<jobid> --pty bash
```

From that shell, check GPU usage:

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

Check that Jupyter is running:

```bash
ps -ef | grep -E "jupyter-lab|jupyter-notebook" | grep -v grep
```

Check whether the port is listening:

```bash
ss -tulnp | grep 8888
```

### 7. Stop the session

When the training is finished, stop the notebook from the Jupyter interface or cancel the Slurm job:

```bash
scancel <jobid>
```

Confirm that it disappeared from the queue:

```bash
squeue -u $USER
```

If you need to cancel all your jobs, use this carefully:

```bash
scancel -u $USER
```
