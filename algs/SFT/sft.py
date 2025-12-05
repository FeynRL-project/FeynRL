import torch
import numpy as np

class SFT:
    def __init__(self,
                model_engine,
                optimizer,
                micro_batch_size_per_gpu,
                clip_grad_norm=None,
                use_cache=False, 
                device='cpu'):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu
        self.use_cache = use_cache
        self.device = device

        # use cross entropy loss
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # clip gradient if required
        if clip_grad_norm is not None:
            self.clip_grad = torch.nn.utils.clip_grad_norm_
        else:
            self.clip_grad = None

    def compute_loss(self, logits, y, mask):
        '''
         This functions implements \sum_{i=1}^{N} log p(y_i|x_i)
         y is target label [B, T -1]
         logits is model prediction [B, T -1, vocab_size]
        '''
        B, T, vocab_size = logits.shape

        # flatten logits across batch and seq_len before computing loss
        # so logits is [B * (T -1), vocab_size]
        logits = logits.view(-1, vocab_size)
        # flatten y as well:  [B, T -1] -->  [B * (T -1)]
        y = y.view(-1)

        # per token loss
        per_token_loss = self.loss_fn(logits, y)

        # We need to apply mask to loss to remove any things 
        # which should not be considered in loss (e.g., padding tokens)
        masked_loss = per_token_loss * mask

        # To avoid gardient accumulation error caused by loss.mean(),
        # we use sum of loss instead but play with learning rate to account for this.
        loss = masked_loss.sum()
        return loss

    def forward(self, batch):
        '''
            This function implements a single forward pass for current batch.
            It returns logits, y, and mask
            logits: [B, T -1, vocab_size]
            y: [B, T -1]
            mask: [B, T -1]
        '''
        # batch is a dictionary, so we want to extract things we need from it
        # input_ids/att_mask/pos_ids are [B, T]
        input_ids = batch['input_ids'].to(self.device)
        att_mask  = batch['attention_mask'].to(self.device)
        pos_ids   = batch['position_ids'].to(self.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, vocab_size]
        every_token_logits = output.logits

        # label is input_ids shifted by one
        # so y is [B, T -1]
        y = input_ids[:, 1:].contiguous()
        # as a result we need to remove last token from every_token_logits
        # so logits is [B, T -1, vocab_size]
        logits = every_token_logits[:, :-1, :].contiguous()

        # last input token is EOS, so we ignore loss theal  e. so mask is [B, T -1]
        mask = batch['loss_mask'][:, :-1].contiguous()

        return logits, y, mask

    def val_step(self, data):
        '''
           This function implements a single validation step per rank/gpu.
        '''
        # we need to split data into micro batches
        micro_batches = data.split(self.micro_batch_size_per_gpu)
        num_of_micro_batches = len(micro_batches)
        val_loss = 0
        self.model_engine.eval()
        with torch.no_grad():
            for batch in micro_batches:
                ######## 
                # forward pass per gpu/rank
                ########
                logits, y, mask = self.forward(batch)

                ######## 
                # compute loss pass
                ########
                loss = self.compute_loss(logits=logits, y=y, mask=mask)

                val_loss += loss.item()/num_of_micro_batches

        return val_loss

    def train_step(self, data):
        '''
           This function implements a single training step per rank/gpu.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu. 
           So we need to split data into micro batches.  
        '''
        # we need to split data into micro batches
        micro_batches = data.split(self.micro_batch_size_per_gpu)
        num_of_micro_batches = len(micro_batches)
        step_loss = 0
        # make sure model is in training mode
        self.model_engine.train()
        for batch in micro_batches:
            ######## 
            # forward pass per gpu/rank
            ########
            logits, y, mask = self.forward(batch)

            ######## 
            # compute loss pass
            ########
            loss = self.compute_loss(logits=logits, y=y, mask=mask)

            ########    
            # backward step
            ########
            self.optimizer.zero_grad()
            loss = loss / num_of_micro_batches            
            loss.backward()
            step_loss += loss.item()

        self.model_engine.step()
        return step_loss