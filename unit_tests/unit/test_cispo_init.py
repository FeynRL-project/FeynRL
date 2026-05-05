import torch
import pytest
from unittest.mock import MagicMock, patch
from algs.CISPO.cispo import CISPO

def test_cispo_init_and_engine():
    # Mocking arguments
    model_path = "mock/model"
    deepspeed_config = MagicMock()
    deepspeed_config.model_dump.return_value = {}
    
    # We need to patch load_model to avoid HF calls
    with patch.object(CISPO, 'load_model') as mock_load:
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
        
        cispo = CISPO(
            model_path=model_path,
            model_dtype=torch.float32,
            trust_remote_code=True,
            attn_impl="",
            kl_coeff=0.01,
            clip_low=0.2,
            clip_high=0.2,
            entropy_coeff=0.01,
            micro_batch_size_per_gpu=1,
            update_after_full_replay=True,
            normalize_loss=False,
            deepspeed_config=deepspeed_config,
            gradient_checkpointing=False,
            seed=42,
            train_steps_per_epoch=1,
        )
        
        assert cispo.ready is True
        assert cispo.alg_name == "CISPO"
        assert deepspeed.initialize.call_count >= 1 # policy
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
        cispo.policy_forward = MagicMock(return_value=(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3)))
        cispo.compute_policy_loss = MagicMock(return_value=(torch.tensor(1.0, requires_grad=True), torch.tensor(1.0), {'clipfrac': 0.1, 'approx_kl': 0.01, 'kl_ref': 0.0, 'ent_loss': 0.0, 'pi_loss': 1.0, 'loss_total': 1.0}))
        
        # Setup engine mocks
        cispo.policy_engine.device = torch.device('cpu')
        cispo.policy_engine.gradient_accumulation_steps = MagicMock(return_value=1)
        
        metrics = cispo.train_step(engine_id=0, micro_batches=micro_batches)
        
        assert 'pi_loss' in metrics
        assert cispo.policy_engine.backward.called
