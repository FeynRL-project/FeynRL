import torch
import pytest
from unittest.mock import MagicMock, patch
from algs.DAPO.dapo import DAPO

def test_dapo_init_and_engine():
    # Mocking arguments
    model_path = "mock/model"
    deepspeed_config = MagicMock()
    deepspeed_config.model_dump.return_value = {}

    # We need to patch load_model to avoid HF calls
    with patch.object(DAPO, 'load_model') as mock_load:
        # Mocking values returned by load_model
        policy_model = MagicMock(spec=torch.nn.Module)
        policy_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1), requires_grad=True)]

        mock_load.return_value = {
            "policy_model": policy_model,
            "ref_model": None,
        }

        # We also need to mock deepspeed.initialize
        import deepspeed
        deepspeed.initialize.return_value = (MagicMock(), MagicMock(), None, None)

        dapo = DAPO(
            model_path=model_path,
            model_dtype=torch.float32,
            trust_remote_code=True,
            attn_impl="",
            kl_coeff=0.0,
            clip_low=0.2,
            clip_high=0.28,  # clip-higher: the DAPO-defining asymmetry
            entropy_coeff=0.0,
            micro_batch_size_per_gpu=1,
            update_after_full_replay=True,
            normalize_loss=True,
            deepspeed_config=deepspeed_config,
            gradient_checkpointing=False,
            seed=42,
            train_steps_per_epoch=1,
        )

        assert dapo.ready is True
        assert dapo.alg_name == "DAPO"
        # asymmetric clip range must be stored as-is
        assert dapo.clip_low == 0.2
        assert dapo.clip_high == 0.28
        assert deepspeed.initialize.call_count >= 1  # policy
        mock_load.assert_called_once()

        # Now test train_step
        micro_batches = [
            {
                'input_ids': torch.zeros(1, 4, dtype=torch.long),
                'attn_mask': torch.ones(1, 4),
                'zscore': torch.zeros(1, 4),
                'mask': torch.ones(1, 4),
                'old_logprobs': torch.zeros(1, 4),
            }
        ]

        # Mock forward/loss methods called inside train_step
        dapo.policy_forward = MagicMock(return_value=(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3)))
        dapo.compute_policy_loss = MagicMock(return_value=(torch.tensor(1.0, requires_grad=True), torch.tensor(1.0), {'clipfrac': 0.1, 'approx_kl': 0.01, 'kl_ref': 0.0, 'ent_mc': 0.0, 'pi_loss': 1.0, 'loss_total': 1.0}))
        # normalize_loss=True path computes the global token denominator
        dapo.compute_global_token_denom = MagicMock(return_value=(3.0, 1.0))

        # Setup engine mocks
        dapo.policy_engine.device = torch.device('cpu')
        dapo.policy_engine.gradient_accumulation_steps = MagicMock(return_value=1)

        metrics = dapo.train_step(engine_id=0, micro_batches=micro_batches)

        assert 'pi_loss' in metrics
        assert dapo.policy_engine.backward.called
        # model_class defaults to "llm" when not supplied (back-compat).
        assert dapo.model_class == "llm"


def test_dapo_init_async_overlap_mode():
    '''In overlap (async) mode create_training_engines passes
    use_decoupled_loss=True and behave_imp_weight_cap; DAPO must accept both
    and run train_step through the decoupled (prox-snapshot) path.'''
    deepspeed_config = MagicMock()
    deepspeed_config.model_dump.return_value = {}

    with patch.object(DAPO, 'load_model') as mock_load:
        policy_model = MagicMock(spec=torch.nn.Module)
        policy_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1), requires_grad=True)]
        mock_load.return_value = {"policy_model": policy_model, "ref_model": None}

        import deepspeed
        deepspeed.initialize.return_value = (MagicMock(), MagicMock(), None, None)

        dapo = DAPO(
            model_path="mock/model", model_dtype=torch.float32, trust_remote_code=True,
            attn_impl="eager", kl_coeff=0.0, clip_low=0.2, clip_high=0.28, entropy_coeff=0.0,
            micro_batch_size_per_gpu=1, update_after_full_replay=True, normalize_loss=True,
            deepspeed_config=deepspeed_config, gradient_checkpointing=False, seed=42,
            train_steps_per_epoch=1,
            use_decoupled_loss=True, behave_imp_weight_cap=2.0,
        )
        assert dapo.use_decoupled_loss is True
        assert dapo.behave_imp_weight_cap == 2.0

        micro_batches = [
            {
                'input_ids': torch.zeros(1, 4, dtype=torch.long),
                'attn_mask': torch.ones(1, 4),
                'zscore': torch.zeros(1, 4),
                'mask': torch.ones(1, 4),
                'old_logprobs': torch.zeros(1, 4),
            }
        ]

        # Decoupled path snapshots pi_prox once per epoch; mock it out.
        prox_lp = [torch.zeros(1, 3)]
        prox_nan = [torch.zeros(1, 3, dtype=torch.bool)]
        dapo.snapshot_prox_for_epoch = MagicMock(return_value=(micro_batches, prox_lp, prox_nan))
        dapo.release_prox_cache_if_epoch_end = MagicMock()
        dapo.policy_forward = MagicMock(return_value=(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3)))
        dapo.compute_policy_loss = MagicMock(return_value=(torch.tensor(1.0, requires_grad=True), torch.tensor(1.0), {'clipfrac': 0.0, 'approx_kl': 0.0, 'kl_ref': 0.0, 'ent_mc': 0.0, 'pi_loss': 1.0, 'loss_total': 1.0}))
        dapo.compute_global_token_denom = MagicMock(return_value=(3.0, 1.0))
        dapo.policy_engine.device = torch.device('cpu')
        dapo.policy_engine.gradient_accumulation_steps = MagicMock(return_value=1)

        metrics = dapo.train_step(engine_id=0, micro_batches=micro_batches)

        assert 'pi_loss' in metrics
        dapo.snapshot_prox_for_epoch.assert_called_once()
        # the prox snapshot must be forwarded into the loss
        _, kwargs = dapo.compute_policy_loss.call_args
        assert kwargs['prox_logprobs'] is prox_lp[0]