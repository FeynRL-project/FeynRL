## It seems this framework uses a less efficient setup where the rollout engine and training engine are not overlapped, which can lead to GPU underutilization. Why not use parallel engines to increase rollout generation throughput?

That is a good point. In the first few iterations of this work, we mainly prioritized the algorithmic side over maximizing system throughput. The main difficulty in RL is not that rollout generation takes time. It is that we still do not have RL algorithms that reliably work well for large models, especially when rewards are sparse, reward signals are noisy or misspecified, and training can be brittle to small details. If we do not yet have a reliable method that can learn an “optimal” policy when it should be possible, increasing rollout throughput will not necessarily improve the situation, and may simply add unnecessary complexity. If you care about moving fast, debugging quickly, understanding how the underlying system and algorithms work, and extending methods with confidence, while still being able to train large models at scale in a production grade setting,, LeanRL is built for you.

This does not imply that system throughput is unimportant or can be ignored. It emphasizes that RL itself has many fundamental challenges, and system optimization pays off most once the algorithm is in a healthy place. That said, we do plan to adopt proven patterns from other works, such as pipelined rollout and training overlap, as long as we can do it without much sacrificing the core goals of this repo.


## There are differences between your implementation of methods like GRPO. Why is that the case?

That is correct. RL training is sensitive to small implementation details, and some important details are often overlooked when applying RL to large models, unlike classic RL settings such as games. As a result, in some places LeanRL makes deliberate choices to improve stability and performance, even if that means it does not match a specific reference implementation line for line.

When the differences are intentional, we document them and name variants explicitly. For example, you may see SGRPO, which indicates a GRPO style method with stability focused implementation choices and some clear changes from the original work.

## I found a bug. What should I do?

That is wonderful. Please open a GitHub issue with steps to reproduce, expected behavior, and actual behavior. If you can include logs, config files, or a minimal script, it will help a lot. Pull requests are also welcome, and we will review them as quickly as possible.

## I have a few research ideas and want guidance. Can you help?

We can try. If you are comfortable sharing your idea publicly, open a GitHub issue and include enough context for others to follow along. If you prefer to discuss privately, you can email Rasool. Contact details are available on the [Rasool's website](https://rasoolfa.github.io/).

