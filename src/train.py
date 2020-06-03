import config
import torch
import dataset
import engine
import numpy as np
import pandas as pd
import torch.nn as nn
from apex import amp
from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup



def run(opt_level="O2", keep_batchnorm_fp32=True, batch_size=5, nb_epochs=10, data_path="../inputs/IMDB_Dataset.csv"):
    #df = pd.read_csv(config.TRAINING_FILE).fillna("none")[0:100] # Essai pour ne pas prendre trop de temps
    df = pd.read_csv(data_path).fillna("none")[0:100]
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    
    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df.sentiment.values 
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    # Creating the datasets
    train_dataset = dataset.BERTDataset(
        review=df_train.review.values,
        target=df_train.sentiment.values
    )
    
    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values,
        target=df_valid.sentiment.values
    )    
    # Creating the dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,#=config.TRAIN_BATCH_SIZE,
        num_workers=10,
        drop_last=True
    )
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size,#=config.VALID_BATCH_SIZE,
        num_workers=10,
        drop_last=True
    )    
    # Defining the model and sending to the device
    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)
    
    parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"] # We don't want any decay for them
    optimizer_parameters = [
    {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
    {"params": [p for n, p in parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ] 
    #nb_epochs = config.EPOCHS
    num_train_steps = int(len(df_train) * nb_epochs / batch_size)#config.TRAIN_BATCH_SIZE)
    # Defining the optimizer and the scheduler
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
 
    #model = nn.DataParallel(model) # If you have multi-gpus
    # Initialize the pytorch model and the optimizer to allow mixed-precision training
    model, optimizer = amp.initialize(model, optimizer, 
                                      opt_level=opt_level,
                                      keep_batchnorm_fp32=keep_batchnorm_fp32, 
                                      loss_scale="dynamic")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # No warmup
        num_training_steps=num_train_steps
    )
    
    # Train the model
    engine.global_trainer(train_dataloader, valid_dataloader, model, optimizer, scheduler, device, nb_epochs)
    
if __name__ == "__main__":
    run()