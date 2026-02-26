# FAQ
## Why this repo called FeynRL?

FeynRL (pronounced like “FineRL”) nods to Richard Feynman’s emphasis on clear, computational thinking, which is exactly what this repository aims for in building RL methods. It also loosely echoes his "sum over histories" view in quantum mechanics, where certain predictions are computed by summing contributions from many possible paths; likewise, RL improves by learning from many sampled trajectories (rollouts), not a single one.

## Why should I use this framework?

This is an **RL-first** repo that focuses heavily on the algorithmic and research side of RL for large models, not just infrastructure. We also discuss and implement practical methods and training tricks that are commonly used in training frontier models but are rarely written down publicly.

More importantly, FeynRL is designed to remove a common trade off. Many industry-grade frameworks can train at scale, but they are hard to modify and not ideal for doing research. Many research frameworks are easy to change, but they typically only support toy settings, and it's unclear how well those results translate to realistic scenarios.

Our goal is to build something that lets you do both. You should be able to run realistic experiments on large data with multi node, multi GPU training, while still understanding what is happening, debugging quickly, and making changes with confidence. If you care about moving fast, understanding how the underlying system and algorithms work, and extending methods without losing the ability to train at scale in a production grade setting, FeynRL is built for you.

## Why not run rollout engines fully in parallel (continuous generation) while training runs?

That’s a good point; not overlapping rollout and training can leave some GPU capacity unused. The reason this framework doesn't default to "always-on" rollout is mainly about data off-policyness and algorithmic bottlenecks, not just throughput.

1. **On-policy methods don't benefit much from stale, continuously generated data.** Most practical RL post-training recipes for large models are effectively "on-policy" (or close to it) and rely on mechanisms like PPO-style clipping (or related constraints) to stay stable when the policy changes. If a rollout engine keeps generating while the policy is being updated, a growing fraction of those samples quickly become off-policy. Once the divergence is large, clipping/constraints tend to squash the update signal and many samples contribute little to no useful gradient. In other words, "more generations" is not automatically "more learning" if those generations are produced by a policy that is already out of date relative to the current optimizer state.

2. **The limiting factor is usually algorithm reliability, not raw rollout speed.** In practice, the hard part of RL for large models isn’t only that generation is expensive, it’s the underlying algorithmic limitations. If the underlying method isn’t reliably improving the policy when it should, increasing rollout throughput often just increases complexity (queues, buffering, off-policy correction, synchronization) without improving outcomes.

This does not imply that system throughput is unimportant or can be ignored. It emphasizes that RL itself has many fundamental challenges, and system optimization pays off most once the algorithm is in a healthy place. That said, we do plan to adopt proven patterns from other works as long as we can do it without much sacrificing the core goals of this repo.

## Other frameworks include many rollout-engine system improvements. Why don't you include them?

We'll try to include recent improvements as much as possible, especially with regard to the rollout engine—and this is one of the reasons we open-sourced the repo.

That said, some of these improvements have only a marginal impact on performance, but add significant complexity to the pipeline (see response to previous question). In cases like that, we avoid including them by default. However, we are open to PRs that improve system throughput without sacrificing the core goals of this repo.

## There are differences between your implementation of methods like GRPO. Why is that the case?

That is correct. RL training is sensitive to small implementation details, and some important details are often overlooked when applying RL to large models, unlike classic RL settings such as games. As a result, in some places FeynRL makes deliberate choices to improve stability and performance, even if that means it does not match a specific reference implementation line for line.

When the differences are intentional, we document them and name variants explicitly. For example, you may see SGRPO, which indicates a GRPO style method with stability focused implementation choices and some clear changes from the original work.

## I found a bug. What should I do?

That is wonderful. Please open a GitHub issue with steps to reproduce, expected behavior, and actual behavior. If you can include logs, config files, or a minimal script, it will help a lot. Pull requests are also welcome, and we will review them as quickly as possible.

## We have a new work where we propose a method (RL or non-RL) that can improve results significantly. Would you be open to adding our method to the repo?

Absolutely. We're open to contributions that improve the repo. Please submit a PR and include enough context (paper link, a short summary, expected gains, and how to reproduce) so others can follow along and review it. Also, please make sure your code is clean and closely follows the repo structure. If you prefer to discuss privately first, you can [email Rasool](https://rasoolfa.github.io/).

## What hardware do I need to run FeynRL?

We have tested FeynRL on NVIDIA A100 and H100 GPUs. That said, any GPU with CUDA support should work as long as you can install the required packages (PyTorch, DeepSpeed, vLLM, etc.). The main constraint is GPU memory: larger models need more VRAM, and you can use DeepSpeed ZeRO Stage 3 with CPU offloading to reduce the per-GPU memory footprint.

## We have a new rollout engine that can significantly improve rollout throughput. Would you be open to adding our rollout engine to the repo?

Of course. Please submit a PR, or if you prefer to discuss it first, you can [email Rasool](https://rasoolfa.github.io/).

## I have a few research ideas and want guidance. Can you help?

We can try. If you are comfortable sharing your idea publicly, open a GitHub issue and include enough context for others to follow along. If you prefer to discuss privately, you can [email Rasool](https://rasoolfa.github.io/).
## I think there are issues in the code and I have ideas to help improve it. What should I do?

We're happy to hear that—this is exactly why we open-sourced the project. Please submit a GitHub issue describing what you found and any suggestions you have. Pull requests are also very welcome.

## I have a question that's not covered here. Where can I ask?

Please open a GitHub issue with your question.
