## How to use

After setting up the environment, the next step is preparing the data. See the scripts in [`data-prep/`](https://github.com/rasoolfa/FeynRL/tree/main/data_prep) for reference implementations—you can adapt them to your own datasets.

**Data format requirement:** your final processed data must match the **exact** format produced by these scripts (the original/raw format does not matter). You need to write your own scripts simailr to the following scripts to prepare your data in the required format.

* [`data-prep/gsm8k.py`](https://github.com/rasoolfa/FeynRL/blob/main/data_prep/gsm8k.py) prepares **GSM8K** in a format suitable for **SFT** and **RL** training, and can also be used for evaluation.
* [`data-prep/hh_rlhf.py`](https://github.com/rasoolfa/FeynRL/blob/main/data_prep/hh_rlhf.py) prepares a **preference/contrastive** dataset suitable for **DPO**-style contrastive learning.

Once your data is prepared, update the **`data`** section in the relevant config file and run the corresponding entrypoint:

* SFT: `./configs/sl_args.yaml`
* Contrastive Learning (DPO, etc.): `./configs/cl_args.yaml`
* RL: `./configs/rl_args.yaml`

---

## Running on a single node

“Single node” means one machine with multiple GPUs.

### Supervised Fine-Tuning (SFT)

`main_sl.py` is the entry point for supervised learning experiments.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main_sl.py --config_file ./configs/sl_args.yaml --experiment_id myexp1
```

**Notes**

* `experiment_id` is the name of the experiment. It is used to create an output directory to store logs, checkpoints, and metrics.
* `CUDA_VISIBLE_DEVICES` selects which GPUs are visible to the run.
* `--nproc_per_node` must match the number of visible GPUs (e.g., 4 GPUs ⇒ `--nproc_per_node=4`).
* Before running, review `sl_args.yaml` and update at least:

  * model / tokenizer path
  * dataset path(s)
  * batch sizes / gradient accumulation
  * output directory / logging config
* We call it `main_sl.py` rather than `main_sft.py` because this entry point is intended to support **general supervised learning**, not just SFT.

---

### Contrastive Learning (CL) — DPO and related methods

`main_cl.py` is the entry point for contrastive/preference learning experiments (e.g., DPO).

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_cl.py --config_file ./configs/cl_args.yaml --experiment_id myexp2
```

**Notes**

* Ensure `cl_args.yaml` points to the processed dataset paths and that the expected fields match what the trainer expects.

---

### Reinforcement Learning (RL)

RL runs are more involved because they use **Ray** to orchestrate DeepSpeed **training** and **rollout** engines.

`main_rl.py` is the entry point for RL experiments (e.g., PPO, GRPO, CISPO, etc.).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main_rl.py --config_file ./configs/rl_args.yaml --experiment_id exp3
```

**Key differences vs SFT/CL**

* RL uses two types of engines:

  * **Training engine** (based on DeepSpeed)
  * **Rollout engine(s)** (inference/generation for trajectories which is based on vLLM)
* Ray schedules these workers across available GPUs.

**Config knobs**
In `rl_args.yaml`, make sure you clearly document and set:

* `rollout_gpus`: number of GPUs reserved for rollout workers
* `training_gpus`: number of GPUs reserved for training workers
* `ray_address`:

  * single node: set to `null` to start locally
  * multi node: see next section
* `ray_master_port`: port used by DeepSpeed/NCCL rendezvous.

**Single-node Ray**

* If `ray_address` is `null`, the code should start/connect to Ray on the local machine automatically.
* `ray_master_port` can be any free port on the node (example: `25000`).


---

## Running on multiple nodes

“Multi-node” means Ray spans multiple machines and schedules rollout/training workers across them.

Assume:

* Node A (head): 4 GPUs
* Node B (worker): 4 GPUs

### 1) Start Ray head on the main node

On **Node A**:

```bash
CUDA_VISIBLE_DEVICES=3,5,2,4 ray start --head --port=26789
```

Ray will print the **IP address** of the head node (example: `100.9.128.5`) and connection instructions.

### 2) Join worker nodes

On **Node B**:

```bash
CUDA_VISIBLE_DEVICES=0,1,5,7 ray start --address=100.9.128.5:26789
```

If successful, Ray will report that the node joined the cluster.

### 3) Update RL config

In `./configs/rl_args.yaml`:

* Set `ray_address: "auto"`.
* Set `ray_master_port` to a **different free port** than Ray’s port (example: `25000`).

**Why `ray_master_port` must be different**

* Ray uses `26789` for cluster coordination.
* DeepSpeed/NCCL needs its own rendezvous port (`ray_master_port`) for distributed training setup.

### 4) Run RL from the head node

On **Node A**:

```bash
python main_rl.py --config_file ./configs/rl_args.yaml --experiment_id exp4
```

**Important**

* Do **not** set `CUDA_VISIBLE_DEVICES` for the multi-node RL run. Ray discovers and manages GPUs across nodes; forcing visibility can cause mismatches and scheduling errors.

---

## Running with Docker

A pre-built Docker image is available on Docker Hub with all dependencies installed.

### Pull the image

```bash
docker pull popsodazhp/feyn_rl:latest
```

Or build it yourself:

```bash
docker build -t feyn_rl:latest .
```

### SFT with Docker

```bash
docker run --gpus all --ipc=host --net=host \
  -v /path/to/hf_cache:/root/.cache/huggingface \
  -w /FeynRL \
  popsodazhp/feyn_rl:latest \
  torchrun --nproc_per_node=4 main_sl.py \
    --config_file ./configs/sl_args.yaml \
    --experiment_id myexp1
```

### RL with Docker

```bash
docker run --gpus all --ipc=host --net=host \
  -v /path/to/hf_cache:/root/.cache/huggingface \
  -w /FeynRL \
  popsodazhp/feyn_rl:latest \
  python main_rl.py \
    --config_file ./configs/rl_args.yaml \
    --experiment_id exp3
```

**Notes**

* `--gpus all` exposes all GPUs. Use `--gpus '"device=0,1,2,3"'` to select specific GPUs.
* `--ipc=host` is required for PyTorch shared memory.
* Mount your HuggingFace cache to avoid re-downloading models.
* Mount a volume for checkpoints if you want to persist them: `-v /path/to/ckps:/FeynRL/ckps`

### Slurm (enroot)

Convert the Docker image to an enroot squashfs image:

```bash
enroot import dockerd://popsodazhp/feyn_rl:latest
```

Then use it in your sbatch script with `--container-image`.

---

## Troubleshooting

For troubleshooting common issues related to DeepSpeed, Ray, multi-node scaling, and training stability, please refer to the [Troubleshooting Guide](../docs/TROUBLESHOOTING.md).