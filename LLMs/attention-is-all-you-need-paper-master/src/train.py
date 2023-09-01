import os
from logging import log

import torch
import torch.nn as nn
import torch.optim as optim


# from learner import Learner
from scheduler import CustomScheduler
from dataset import get_translation_dataloaders
from callbacks import CheckpointSaver, MoveToDeviceCallback, TrackLoss, TrackExample, TrackBleu
from architectures.machine_translation_transformer import MachineTranslationTransformer




# Configure Logging
from utils.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Seed the Random Number Generators
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


class Training:
    def __init__(self):

        log.info('----- Training Started -----')
        # Initialize configuration
        self.config=dict(
            # RUN CONFIG:
            RUN_NAME='cpu_run',
            RUN_DESCRIPTION="""
                This run is for testing can the model overfit a single example.
                It is useful when debugging.
                For better results change the scheduler in train.py.
                """,
            RUNS_FOLDER_PTH='LLMs/attention-is-all-you-need-paper-master/runs',
            # DATA CONFIG:
            DATASET_SIZE=2,
            TEST_PROPORTION=0.5,
            MAX_SEQ_LEN=100,
            VOCAB_SIZE=100,
            TOKENIZER_TYPE='wordlevel', # 'wordlevel' or 'bpe
            # TRAINING CONFIG:
            BATCH_SIZE=1, 
            GRAD_ACCUMULATION_STEPS=1,
            WORKER_COUNT=10,
            EPOCHS=1000,
            # OPTIMIZER CONFIG:
            BETAS=(0.9, 0.98),
            EPS=1e-9,
            # SCHEDULER CONFIG:
            N_WARMUP_STEPS=4000,
            # MODEL CONFIG:
            D_MODEL=512,
            N_BLOCKS=6,
            N_HEADS=8,
            D_FF=2048,
            DROPOUT_PROBA=0.1,
            # OTHER:
            MODEL_SAVE_EPOCH_CNT=1000,
            DEVICE='cpu',
            LABEL_SMOOTHING=0.1,
        )
        # Device handling
        if self.config['DEVICE']=='gpu':
            device='cuda'
        else:
            self.device='cpu'

    def main(self):

        train_dl, val_dl = get_translation_dataloaders(
        dataset_size=self.config['DATASET_SIZE'],
        vocab_size=self.config['VOCAB_SIZE'],
        tokenizer_save_pth=os.path.join(self.config['RUNS_FOLDER_PTH'],self.config['RUN_NAME'],'tokenizer.json'),
        tokenizer_type=self.config['TOKENIZER_TYPE'],
        batch_size=self.config['BATCH_SIZE'],
        report_summary=True,
        max_seq_len=self.config['MAX_SEQ_LEN'],
        test_proportion=self.config['TEST_PROPORTION'],
        )

        model = MachineTranslationTransformer(
        d_model=self.config['D_MODEL'],
        n_blocks=self.config['N_BLOCKS'],
        src_vocab_size=self.config['VOCAB_SIZE'],
        trg_vocab_size=self.config['VOCAB_SIZE'],
        n_heads=self.config['N_HEADS'],
        d_ff=self.config['D_FF'],
        dropout_proba=self.config['DROPOUT_PROBA']
        )

        loss_func = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1, reduction='mean')

        optimizer = optim.Adam(model.parameters(), betas=self.config['BETAS'], eps=self.config['EPS'])
        scheduler=CustomScheduler(optimizer, self.config['D_MODEL'], self.config['N_WARMUP_STEPS'])
        
        # # The above scheduler's efficiency is highly influenced by dataset and batch size,
        # # alternatively you can use the below configuration, which also works much better for overfit configs.
        # optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=wandb.config.BETAS, eps=wandb.config.EPS)
        # scheduler=optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, epochs=wandb.config.EPOCHS, steps_per_epoch=len(train_dl), pct_start=0.3)
        
        cur_step=1
        model.to(self.device)
        for epoch_idx in range(self.config['EPOCHS']):
            
            # Train
            train_loss=0
            model.train()
            for batch_idx, batch in enumerate(train_dl):
                xb,yb=batch
                xb,yb =xb.to(self.device),yb.to(self.device)
                preds=model(xb,yb)
                
                loss=loss_func(
                    preds.reshape(-1, preds.size(-1)), # Reshaping for loss
                    yb[:,1:].contiguous().view(-1) # Shifting right (without BOS)
                )
                train_loss+=loss.detach().cpu()
                
                
                loss.backward()   
                if cur_step % self.config['GRAD_ACCUMULATION_STEPS']==0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    cur_step+=1
                
            # Validation
            val_loss=0
            with torch.no_grad():
                model.eval()
                for batch_idx, batch in enumerate(val_dl):
                    xb,yb=batch
                    xb,yb =xb.to(self.device),yb.to(self.device)
                    preds=model(xb,yb)

                    loss=loss_func(
                        preds.reshape(-1, preds.size(-1)), # Reshaping for loss
                        yb[:,1:].contiguous().view(-1) # Shifting right (without BOS)
                    )
                    val_loss+=loss.detach().cpu()
                    
            print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}")


            # Inference: Change the code later
            
            # model.translate(input_sentence, tokenizer)

        
if __name__ == "__main__":
    Training().main()
    
