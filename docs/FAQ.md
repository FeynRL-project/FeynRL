# FAQ
## Why this repo called FeynRL?

FeynRL (pronounced like “FineRL”) nods to Richard Feynman’s emphasis on clear, computational thinking, which is exactly what this repository aims for in building RL methods. It also loosely echoes his "sum over histories" view in quantum mechanics, where certain predictions are computed by summing contributions from many possible paths; likewise, RL improves by learning from many sampled trajectories (rollouts), not a single one.

## Why should I use this framework?

This is an **RL-first** repo that focuses heavily on the algorithmic and research side of RL for large models, not just infrastructure. We also discuss and implement practical methods and training tricks that are commonly used in training frontier models but are rarely written down publicly.

FeynRL is designed to address a common trade-off that we ourselves have experienced. Training at scale and ease of modification don't always go hand in hand: scaling-focused codebases can be hard to change, while research-focused ones sometimes only support toy settings.

Our goal is to make it possible to do both. You should be able to run realistic experiments on large data with multi-node, multi-GPU training, while still understanding what is happening, debugging quickly, and making changes with confidence. If you care about moving fast, understanding how the underlying system and algorithms work, and extending methods without losing the ability to train at scale, FeynRL is designed with that in mind.

## Why not run rollout engines fully in parallel (continuous generation) while training runs?

That's a fair point that not overlapping rollout and training can leave some GPU capacity unused. The reason this framework doesn't default to "always-on" rollout is mainly about data off-policyness and algorithmic bottlenecks, not just throughput.

1. **On-policy methods are sensitive to data freshness.** Most practical RL post-training recipes for large models are effectively on-policy (or close to it) and rely on mechanisms like PPO-style clipping to stay stable when the policy changes. If a rollout engine keeps generating while the policy is being updated, a growing fraction of those samples can become off-policy. Once the divergence is large enough, clipping tends to reduce the update signal, and many samples may contribute less useful gradient. This doesn't mean continuous generation is always wrong, it's a trade-off, and the right choice depends on the setting.

2. **The limiting factor is usually algorithm reliability, not raw rollout speed.** In practice, the hard part of RL for large models isn’t only that generation is expensive, it’s the underlying algorithmic limitations. If the underlying method isn’t reliably improving the policy when it should, increasing rollout throughput often just increases complexity (queues, buffering, off-policy correction, synchronization) without improving outcomes.

This does not imply that system throughput is unimportant or can be ignored. It emphasizes that RL itself has many fundamental challenges, and system optimization pays off most once the algorithm is in a healthy place. That said, we actively develop ways to improve system throughput and plan to adopt proven patterns from other works as they mature, as long as they align with the design goals of this repo.

## Other frameworks include many system improvements. Why don't you include them?

We'll try to include recent improvements as much as possible, especially with regard to the rollout engine, and this is one of the reasons we open-sourced the repo.

That said, some improvements may have a modest impact on end-to-end performance while adding notable complexity to the pipeline. In cases like that, we tend to hold off until the trade-off is clearer. We are always open to PRs that improve system throughput, and we evaluate each on its merits.

## There are differences between your implementation of methods like GRPO. Why is that the case?

That is correct. RL training is sensitive to small implementation details, and some choices that work well in settings like games may need revisiting when applying RL to large models. As a result, FeynRL sometimes makes deliberate implementation choices to improve stability and performance, even if that means it does not match a specific reference implementation line for line. When the differences are intentional, we document them explicitly.

## I found a bug. What should I do?

That is wonderful. Please open a GitHub issue with steps to reproduce, expected behavior, and actual behavior. If you can include logs, config files, or a minimal script, it will help a lot. Pull requests are also welcome, and we will review them as quickly as possible.

## We have a new work where we propose a method (RL or non-RL) that can improve results significantly. Would you be open to adding our method to the repo?

Absolutely. We're open to contributions that improve the repo. Please submit a PR and include enough context (paper link, a short summary, expected gains, and how to reproduce) so others can follow along and review it. Also, please make sure your code is clean and closely follows the repo structure. If you prefer to discuss privately first, you can [email Rasool](https://rasoolfa.github.io/).

## What hardware do I need to run FeynRL?

We have tested FeynRL on NVIDIA A100 and H100 GPUs. That said, any GPU with CUDA support should work as long as you can install the required packages (PyTorch, DeepSpeed, vLLM, etc.). The main constraint is GPU memory: larger models need more VRAM, and you can use DeepSpeed ZeRO Stage 3 with CPU offloading to reduce the per-GPU memory footprint.

## Other frameworks use similar components. What makes FeynRL different?

Many frameworks share similar building blocks and we don't claim otherwise. FeynRL's key differentiator is its focus on data, algorithmic, and system clarity and modularity. Rather than optimizing solely for system performance, we prioritize making algorithms easy to understand, modify, and extend while maintaining comparable performance. This is achieved through clean separation of concerns and a plug-and-play design that makes it straightforward to swap components or experiment with new ideas.


## We have a new rollout engine that can significantly improve rollout throughput. Would you be open to adding our rollout engine to the repo?

Of course. Please submit a PR, or if you prefer to discuss it first, you can [email Rasool](https://rasoolfa.github.io/).

## I have a few research ideas and want guidance. Can you help?

We can try. If you are comfortable sharing your idea publicly, open a GitHub issue and include enough context for others to follow along. If you prefer to discuss privately, you can [email Rasool](https://rasoolfa.github.io/).

## I think there are issues in the code and I have ideas to help improve it. What should I do?

We're happy to hear that—this is exactly why we open-sourced the project. Please submit a GitHub issue describing what you found and any suggestions you have. Pull requests are also very welcome.

## I have a question that's not covered here. Where can I ask?

Please open a GitHub issue with your question.

## I'm having issues with my training run. Where can I find help?

Please refer to our [Troubleshooting Guide](./TROUBLESHOOTING.md) for solutions to common issues related to multi-node scaling, memory management, and training stability.
