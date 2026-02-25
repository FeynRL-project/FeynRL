import torch
import pickle

class WeightSyncExtension:
    '''
        vllm WorkerExtension that enables in-place weight updates on vllm workers.
        Used with worker_extension_cls parameter when creating vllm llm instances in vllm_engine.py.
        This allows updating model weights directly in gpu memory without
        destroying and recreating the vllm engine (no disk I/O).
    '''

    def __init__(self, model_runner):
        self.model_runner = model_runner

    def update_weights(self, serialized_state):
        '''
            Update model weights in-place on this vllm worker.
            vllm's load_weights handles name remapping and tp sharding internally.
            serialized_state: file path to pickled state_dict on /dev/shm,
            or a raw dict for backward compatibility.
        '''
        if isinstance(serialized_state, str):
            with open(serialized_state, 'rb') as f:
                state_dict = pickle.load(f)

        elif isinstance(serialized_state, dict):
            state_dict = serialized_state

        else:
            raise TypeError(f"Unsupported weight payload type: {type(serialized_state)}")

        self.model_runner.model.load_weights(weights=state_dict.items())
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
