import numpy as np

def new_accumulator():
    '''
        Create an empty rollout stats accumulator for collecting per-batch statistics.
    '''
    return {'total_samples_generated': 0,
            'all_rewards': [],
            'all_zscores': [],
            'all_named_rewards': {},   # Dict[str, List[float]] for per-component rewards
            'all_response_lens': [],
            'min_response_len': float('inf'),
            'max_response_len': float('-inf'),
            'total_tokens': 0,
            'total_truncated': 0,
            'total_eos': 0,
            'total_finish_stop': 0,
            'total_prompt_len': 0,
            'prompt_response_groups': {},
            'total_logprob_sum': 0.0,
            'total_logprob_tokens': 0,
           }

def accumulate(acc, stats):
    '''
        Merge one batch's stats from merge_rollout_with_stats into the running accumulator.
    '''
    acc['total_samples_generated'] += stats['total_samples_generated']
    acc['all_rewards'].extend(stats['all_rewards'])
    acc['all_zscores'].extend(stats['all_zscores'])
    for name, values in stats.get('all_named_rewards', {}).items():
        acc['all_named_rewards'].setdefault(name, []).extend(values)
    acc['all_response_lens'].extend(stats['all_response_lens'])
    acc['min_response_len'] = min(acc['min_response_len'], stats['min_response_len'])
    acc['max_response_len'] = max(acc['max_response_len'], stats['max_response_len'])
    acc['total_tokens'] += stats['total_tokens']
    acc['total_truncated'] += stats['total_truncated']
    acc['total_eos'] += stats['total_eos']
    acc['total_finish_stop'] += stats['total_finish_stop']
    acc['total_prompt_len'] += stats['total_prompt_len']
    acc['total_logprob_sum'] += stats['total_logprob_sum']
    acc['total_logprob_tokens'] += stats['total_logprob_tokens']
    for pk, (cnt, resp_set) in stats['prompt_response_groups'].items():
        if pk in acc['prompt_response_groups']:
            acc['prompt_response_groups'][pk][0] += cnt
            acc['prompt_response_groups'][pk][1] |= resp_set
        else:
            acc['prompt_response_groups'][pk] = [cnt, resp_set]

def summarize(acc, rollout_time):
    '''
        Compute final rollout metrics (averages, ratios, etc.) from the accumulated raw stats.
    '''
    total = acc['total_samples_generated']
    if total == 0:
        return {"total_samples_generated": 0, "total_tokens": 0,
                "avg_zscore": 0.0, "zscore_std": 0.0,
                "avg_reward": 0.0, "total_reward": 0.0,
                "reward_std": 0.0, "reward_min": 0.0, "reward_max": 0.0,
                "frac_positive_reward": 0.0,
                "avg_response_len": 0.0, "min_response_len": 0.0,
                "max_response_len": 0.0, "response_len_std": 0.0,
                "truncated_ratio": 0.0, "eos_ratio": 0.0,
                "finish_reason_stop_ratio": 0.0,
                "mean_logprob": 0.0, "avg_prompt_len": 0.0,
                "unique_response_ratio": 0.0, "tokens_per_sec": 0.0,
                "rollout_time": rollout_time}

    reward_arr = np.array(acc['all_rewards'])
    zscore_arr = np.array(acc['all_zscores'])

    if acc['prompt_response_groups']:
        ratios = [len(v[1]) / v[0] for v in acc['prompt_response_groups'].values()]
        unique_response_ratio = sum(ratios) / len(ratios)
    else:
        unique_response_ratio = 0.0

    summary = {"total_samples_generated": total,
               "total_tokens": acc['total_tokens'],
               "avg_zscore": float(np.mean(zscore_arr)),
               "zscore_std": float(np.std(zscore_arr)),
               "avg_reward": float(np.mean(reward_arr)),
               "total_reward": float(np.sum(reward_arr)),
               "reward_std": float(np.std(reward_arr)),
               "reward_min": float(np.min(reward_arr)),
               "reward_max": float(np.max(reward_arr)),
               "frac_positive_reward": float(np.mean(reward_arr > 0)),
               "avg_response_len": float(np.mean(acc['all_response_lens'])),
               "min_response_len": acc['min_response_len'],
               "max_response_len": acc['max_response_len'],
               "response_len_std": float(np.std(acc['all_response_lens'])),
               "truncated_ratio": acc['total_truncated'] / total,
               "eos_ratio": acc['total_eos'] / total,
               "finish_reason_stop_ratio": acc['total_finish_stop'] / total,
               "mean_logprob": acc['total_logprob_sum'] / max(1, acc['total_logprob_tokens']),
               "avg_prompt_len": acc['total_prompt_len'] / total,
               "unique_response_ratio": unique_response_ratio,
               "tokens_per_sec": acc['total_tokens'] / max(1e-6, rollout_time),
               "rollout_time": rollout_time}

    # Per-component reward averages, emitted as reward_<name>_avg
    for name, values in acc.get('all_named_rewards', {}).items():
        if values:
            summary[f"reward_{name}_avg"] = float(np.mean(values))

    return summary