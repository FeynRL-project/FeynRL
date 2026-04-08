# FAQ

## Why is this repo called FeynRL?

FeynRL (pronounced like “FineRL”) nods to Richard Feynman’s emphasis on clear, computational thinking, which is exactly what this repository aims for in building RL methods. It also loosely echoes his "sum over histories" view in quantum mechanics, where certain predictions are computed by summing contributions from many possible paths; likewise, RL improves by learning from many sampled trajectories (rollouts), not a single one.

## How is FeynRL different from other post-training frameworks?

Most frameworks share similar building blocks. The difference is the design priority. FeynRL is optimized first for **clarity, locality of change, and method development**, while still supporting scalable execution. Other frameworks may optimize more aggressively for built-in functionality or system throughput, which are valid trade-offs but can make the codebase harder to modify when you want to build something new.

We built FeynRL around a trade-off we experienced directly: training at scale and ease of modification don't always go together. Our goal is to make it possible to do both.

## Does "algorithm-first" mean toy-scale?

No. "Algorithm-first" describes the design priority, not a limitation to small experiments.

FeynRL supports large-scale training through DeepSpeed, Ray, and vLLM, including multi-GPU and multi-node training, sync and overlap execution, adaptive ESS-based weight synchronization, and multiple weight-sync backends.

The goal is to keep the stack understandable without giving up the ability to run realistic experiments.

## Why not run rollout engines fully in parallel (continuous generation) while training runs?

That's a fair point that not overlapping rollout and training can leave some GPU capacity unused. The reason this framework doesn't default to "always-on" rollout is mainly about data off-policyness and algorithmic bottlenecks, not just throughput.

1. **On-policy methods are sensitive to data freshness.** Most practical RL post-training recipes for large models are effectively on-policy (or close to it) and rely on mechanisms like PPO-style clipping to stay stable when the policy changes. If a rollout engine keeps generating while the policy is being updated, a growing fraction of those samples can become off-policy. Once the divergence is large enough, clipping tends to reduce the update signal, and many samples may contribute less useful gradient. This doesn't mean continuous generation is always wrong, it's a trade-off, and the right choice depends on the setting.

2. **The limiting factor is usually algorithm reliability, not raw rollout speed.** In practice, the hard part of RL for large models isn't only that generation is expensive, it's the underlying algorithmic limitations. If the underlying method isn't reliably improving the policy when it should, increasing rollout throughput often just increases complexity (queues, buffering, off-policy correction, synchronization) without improving outcomes.

That said, FeynRL includes an **overlap engine** that provides a practical middle ground. See the next question for details.

## How does the overlap engine work, and when should I use it?

The overlap engine runs rollout generation and training concurrently within a single epoch. It uses a queue-pull architecture: the driver fills a shared Ray prompt queue at the start of each epoch and rollout engines self-schedule by pulling from it, while the driver drains results between training steps. Mid-epoch weight sync is triggered by ESS (for P3O) or a fixed step interval (for PPO/GRPO/CISPO). For a full description of the mechanisms (queue-pull, pipelined generation, mid-epoch NCCL sync, staleness control, pre-launched next epoch) and guidance on when to use each mode, see the [Architecture Overview](./ARCHITECTURE.md#-trainingrollout-scheduling).

## Other frameworks include many system improvements. Why don't you include them?

We'll try to include recent improvements as much as possible, especially with regard to the rollout engine, and this is one of the reasons we open-sourced the repo.

That said, some improvements may have a modest impact on end-to-end performance while adding notable complexity to the pipeline. In cases like that, we tend to hold off until the trade-off is clearer. We are always open to PRs that improve system throughput, and we evaluate each on its merits.

## There are differences between your implementation of methods like GRPO. Why is that the case?

That is correct. RL training is sensitive to small implementation details, and some choices that work well in settings like games may need revisiting when applying RL to large models. As a result, FeynRL sometimes makes deliberate implementation choices to improve stability and performance, even if that means it does not match a specific reference implementation line for line. When the differences are intentional, we document them explicitly.

## I found a bug or have ideas to improve the code. What should I do?

Wonderful! Please open a GitHub issue with steps to reproduce (for bugs) or a description of what you'd like to improve. Pull requests are also very welcome, and we will review them as quickly as possible.

## Would you be open to adding a new method or rollout engine?

Absolutely. Please submit a PR and include enough context (paper link, a short summary, expected gains, and how to reproduce) so others can follow along and review it. Please make sure your code is clean and closely follows the repo structure. If you prefer to discuss privately first, you can [email Rasool](https://rasoolfa.github.io/).

## I have a few research ideas and want guidance. Can you help?

We can try. If you are comfortable sharing your idea publicly, open a GitHub issue and include enough context for others to follow along. If you prefer to discuss privately, you can [email Rasool](https://rasoolfa.github.io/).

## What hardware do I need to run FeynRL?

We have tested FeynRL on NVIDIA A100 and H100 GPUs. That said, any GPU with CUDA support should work as long as you can install the required packages (PyTorch, DeepSpeed, vLLM, etc.). The main constraint is GPU memory: larger models need more VRAM, and you can use DeepSpeed ZeRO Stage 3 with CPU offloading to reduce the per-GPU memory footprint.

## I'm having issues with my training run. Where can I find help?

Please refer to our [Troubleshooting Guide](./TROUBLESHOOTING.md) for solutions to common issues related to multi-node scaling, memory management, and training stability. For anything not covered there, open a GitHub issue.
