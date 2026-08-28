# REP-Makers Compute Server Guide (`ubuntu-makers`)

This guide explains how to connect to the shared compute server, transfer input files, write Slurm job requests, and retrieve job outputs.

---

## 1. Connecting to the Server

All connections to the server are secured and routed through Tailscale.

1. **Install and Run Tailscale:**
   Ensure Tailscale is running on your machine and that you are logged in using your school credentials.
   ```bash
   tailscale up
   tailscale status # Confirm "ubuntu-makers" appears in the active node list
   ```
2. **SSH to the Server:**
   Connect using your assigned username. No password or custom SSH key is needed (Tailscale identity handles authentication):
   ```bash
   ssh USERNAME@ubuntu-makers
   ```

---

## 2. Transferring Files (Inputs & Outputs)

You can move files and datasets between your local machine and the server using the following methods:

### Method A: Git (Best for Code)
1. Commit and push your local code changes to a Git repository (e.g., GitHub).
2. On `ubuntu-makers`, clone or pull the repository:
   ```bash
   git clone <repo-url>
   # or
   git pull
   ```

### Method B: command-line tools (`scp` / `rsync`)
Run these commands from your **local machine**, not from inside the server.

* **Uploading local files/folders (Inputs):**
  ```bash
  # Upload a single script to your home directory (~/)
  scp my_script.py USERNAME@ubuntu-makers:~/

  # Upload a dataset folder to your scratch directory
  rsync -avz /path/to/local/dataset USERNAME@ubuntu-makers:/scratch/USERNAME/
  ```

* **Downloading files/folders (Outputs):**
  ```bash
  # Download a model checkpoint or result file to your local directory
  scp USERNAME@ubuntu-makers:/scratch/USERNAME/checkpoint.pt ./

  # Download an entire results folder
  rsync -avz USERNAME@ubuntu-makers:/scratch/USERNAME/output_logs/ ./local_logs/
  ```

### Method C: VS Code Remote SSH (Easiest GUI)
1. Install the **Remote - SSH** extension in VS Code.
2. Open the Command Palette (`Cmd+Shift+P` on Mac or `Ctrl+Shift+P` on Windows) and select **Remote-SSH: Connect to Host...**
3. Enter `USERNAME@ubuntu-makers`.
4. Once connected, you can browse folders, edit code directly on the server, and drag-and-drop files between your local file explorer and the VS Code window to upload/download them.

---

## 3. Writing Slurm Job Requests

To keep the server responsive, **never** run heavy training, builds, or simulations directly on the login shell. You must submit these tasks to the Slurm scheduler.

### Partitions Quick Reference

| Partition | Time Limit | GPU Available? | Primary Use Case |
|---|---|---|---|
| `cpu` (default) | 2h (Max 24h) | No | Code compilation, builds, Vivado synthesis |
| `gpu` | 2h (Max 24h) | Yes (RTX 4090) | Deep Learning training, CUDA execution |
| `interactive` | 1h (Max 4h) | Yes (RTX 4090) | Interactive shells, rapid testing, debugging |

> [!IMPORTANT]
> Without specifying `--gres=gpu:1`, the GPU will be completely invisible to your job processes.

### Job Requests: Shell Execution (`srun`)

Use `srun` for tasks that run in your terminal session or when debugging.

* **CPU-Only Job:**
  ```bash
  srun -p cpu -c 4 --mem=8G python script.py
  ```
  *(Allocates 4 CPU cores and 8 GB of memory in the `cpu` partition)*

* **GPU Job (RTX 4090):**
  ```bash
  srun -p gpu --gres=gpu:1 -c 4 --mem=16G python train.py
  ```

* **Interactive Debugging Session:**
  If you want an interactive bash terminal inside a GPU allocation:
  ```bash
  srun -p interactive --gres=gpu:1 -c 4 --mem=8G --pty bash
  ```

### Job Requests: Batch Submission (`sbatch`)

For long-running tasks, wrap the run in a bash script and submit it using `sbatch`. This allows you to disconnect from SSH while the job runs in the background.

Create a file named `submit_job.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=my_train_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00                 # Max wall time (HH:MM:SS)
#SBATCH --output=/scratch/%u/logs/%j.out # File for standard output (%u=user, %j=job_id)
#SBATCH --error=/scratch/%u/logs/%j.err  # File for standard error

# 1. Load your required toolchains (Modules do not auto-inherit from your login session)
module load cuda/13.1

# 2. Activate Python environment (from scratch)
source /scratch/$USER/myproject/bin/activate

# 3. Run the workload
python train.py --epochs 50 --data-dir /scratch/$USER/dataset
```

Submit it:
```bash
sbatch submit_job.sh
```

---

## 4. Retrieving and Monitoring Job Outputs

* **Monitor Live Status:**
  ```bash
  squeue -u $USER          # Check status of your queued/running jobs
  sinfo                    # Check partition loads and general nodes info
  ```

* **Cancel a Running/Queued Job:**
  ```bash
  scancel <job_id>
  ```

* **Retrieving console output:**
  If you defined `#SBATCH --output` inside your batch file, Slurm will write all terminal print statements to that file (e.g., `/scratch/USERNAME/logs/12345.out`). You can read it live using:
  ```bash
  tail -f /scratch/USERNAME/logs/12345.out
  ```

* **Retrieving generated files:**
  Any output files (models, plots, logs) written by your scripts to `/scratch/USERNAME/` can be downloaded using `scp`/`rsync` or the VS Code Remote extension as detailed in **Section 2**.
