# Lab 00: Setup

## Goal

Get a working environment for every other lab: your GPU visible inside
WSL2, Python with PyTorch using that GPU, and Docker able to start
GPU containers.

## What you will see

A short report from `check_env.py` that names your GPU, shows its
memory, and times the same matrix multiplication on the CPU and the GPU.
The GPU should be dozens of times faster.

## Time and hardware

About 30 minutes, most of it downloads. Windows with WSL2 and an NVIDIA
GPU.

## Read first

Nothing required. If you want to know why a GPU helps so much, matrix
multiplication is explained in
[Module 1](../../01-vectors-and-dot-products.md).

## Steps

### 1. Update the Windows driver

The GPU driver lives on **Windows**, not in Linux. Update it with the
NVIDIA App or from NVIDIA's driver download page, then restart.

Do **not** install an NVIDIA driver inside WSL2. Windows shares its
driver with WSL automatically, and a Linux driver installed inside WSL
breaks that.

### 2. Update WSL

In **Windows PowerShell**:

```powershell
wsl --update
wsl --shutdown
```

Then open your Ubuntu terminal again.

### 3. Check the GPU from inside WSL

```bash
nvidia-smi
```

You should see a table naming your GPU and showing its memory, for
example `12282MiB`. If the command is not found or shows no GPU, go back
to step 1.

It is normal for the "Processes" part to stay empty under WSL, even
while the GPU is busy. Watch the memory figure instead.

### 4. Turn on systemd

Lab 02 installs a background service, which needs systemd. Check
whether it is already running:

```bash
ps -p 1 -o comm=
```

If this prints `systemd`, go to step 5. Otherwise, add this to
`/etc/wsl.conf` (create the file if it is missing):

```ini
[boot]
systemd=true
```

```bash
sudo nano /etc/wsl.conf
```

Then run `wsl --shutdown` in PowerShell, reopen Ubuntu, and check again.

### 5. Install Python tools

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv git
```

Get the course onto your machine, inside the Linux file system. Files
under `/mnt/c/` are much slower to read from WSL.

```bash
cd ~
git clone https://github.com/\
jpalmer37/agent-skills.git
cd agent-skills/learning/labs
```

### 6. Create the shared environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install \
  -r 00-setup/requirements.txt
```

This installs PyTorch with its own copy of CUDA. You do not need to
install the CUDA toolkit separately.

### 7. Run the check

```bash
python 00-setup/check_env.py
```

A healthy result looks like this. Your numbers will differ.

```text
Python        3.12.3
Running in    WSL2
PyTorch       2.15.0
CUDA ready    yes
GPU           NVIDIA GeForce RTX 4070
GPU memory    12.0 GB
CPU matmul    0.3 TFLOP/s
GPU matmul    25.1 TFLOP/s
GPU speedup   about 80x
```

### 8. Install Docker with GPU support

Lab 03 uses Docker. The simplest route on Windows is **Docker
Desktop**, which passes your GPU into containers with no extra setup.

1. Install Docker Desktop for Windows.
2. In its settings, make sure **Use the WSL 2 based engine** is on.
3. Under **Resources, WSL integration**, turn on your Ubuntu distro.

Then, in Ubuntu, check that a container can see the GPU:

```bash
IMG=nvcr.io/nvidia/k8s/cuda-sample
docker run --rm --gpus=all \
  $IMG:nbody nbody -gpu -benchmark
```

It should name your GPU and report a speed in GFLOP/s.

**Alternative:** if you would rather not use Docker Desktop, you can
install Docker Engine inside Ubuntu plus the NVIDIA Container Toolkit.
Follow NVIDIA's *Installing the NVIDIA Container Toolkit* guide, then
run `sudo nvidia-ctk runtime configure --runtime=docker` and restart
Docker. Use one approach, not both.

### 9. Give WSL enough memory (optional)

WSL uses up to half of your PC's memory by default. Labs 03 and 05 are
happier with more. In Windows, create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=24GB
```

Choose a figure that leaves Windows at least 8 GB. Then run
`wsl --shutdown` and reopen Ubuntu.

## What to notice

- **The speedup.** The GPU result is many times the CPU result for the
  same arithmetic. Training and serving models is mostly matrix
  multiplication, so this ratio is the whole reason GPUs matter here.
- **The memory figure.** About 12 GB is the budget every later lab
  plans around. Some of it is already used by Windows for your
  display, which is why later labs never ask for all of it.

## Spin down and clean up

Nothing is running yet. Keep `.venv` for the other labs. When you are
finished with every lab, the last step of the course is:

```bash
cd ~/agent-skills/learning/labs
rm -rf .venv
```

Docker Desktop can be closed from the Windows system tray when you are
not using it. It reserves memory while it runs.

## Troubleshooting

**`nvidia-smi` not found in WSL.** Update the Windows driver and run
`wsl --update`. Check that you are on WSL2, not WSL1: in PowerShell,
`wsl -l -v` should show version 2.

**`CUDA ready: no` but `nvidia-smi` works.** PyTorch needs a newer
driver than the one installed. Update the Windows driver again. Also
check that you activated the `.venv`.

**`docker: command not found` in Ubuntu.** Turn on WSL integration for
your distro in Docker Desktop's settings, then reopen the terminal.

**`could not select device driver "" with capabilities: [[gpu]]`.**
Docker cannot see the GPU. With Docker Desktop, update it and check the
WSL 2 engine is on. With Docker Engine, you need the NVIDIA Container
Toolkit.

**Everything is slow.** Make sure the course is in your Linux home
folder (`~`), not under `/mnt/c/`.

---

[Labs home](../README.md) ·
[Next: Lab 01, Train a tiny GPT →](../01-train-a-tiny-gpt/README.md)
