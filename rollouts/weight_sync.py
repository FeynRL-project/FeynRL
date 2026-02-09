import torch

class WeightSyncExtension:
    '''
        vllm WorkerExtension that enables in-place weight updates on vllm workers.
        Used with worker_extension_cls parameter when creating vllm llm instances in vllm_engine.py.
        This allows updating model weights directly in gpu memory without
        destroying and recreating the vllm engine (no disk I/O).
    '''


    def __init__(self, model_runner):
        self.model_runner = model_runner

    def update_weights(self, weights):
        '''
            Update model weights in-place on this vllm worker.
            vllm's load_weights handles name remapping and tp sharding internally.
            weights: list of (name, tensor) tuples with huggingface parameter names.
        '''
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

    def check_weights_hash(self, param_name):
        '''
            Return a hash of a specific parameter for verification.
            Useful for confirming weights were updated correctly.
            param_name: name of the parameter to hash.
        '''
        for name, param in self.model_runner.model.named_parameters():
            if name == param_name or param_name in name:
                return param.data.float().sum().item()
        return None
