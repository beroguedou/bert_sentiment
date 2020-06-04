import torch
import dataset
import engine
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from apex import amp
from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


parser = argparse.ArgumentParser()
parser.add_argument("-o","--opt_level", help="Define the optimization level", default="O2")
parser.add_argument("-k", "--keep_batchnorm_fp32", help="Tells to keep batchnorm in FP32 or Note", default=True)
parser.add_argument("-b", "--batch_size", type=int, help="The batch size for the training.", default=5)
parser.add_argument("-e", "--nb_epochs", type=int, help="Number of epochs for training.")
parser.add_argument("-d", "--data_path", help="The path where the csv is.", default="../inputs/IMDB_Dataset.csv")
parser.add_argument("-m", "--model_path", help="The path to save the model", default="./")


def run(opt_level="O2", keep_batchnorm_fp32=True, batch_size=5,
        nb_epochs=10, data_path="../inputs/IMDB_Dataset.csv", model_path="./"):
    
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
        batch_size,
        num_workers=10,
        drop_last=True
    )
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size,
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
    {"params": [p for n, p in parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0}] 
    
    num_train_steps = int(len(df_train) * nb_epochs / batch_size)
    # Defining the optimizer and the scheduler
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    # Initialize the pytorch model and the optimizer to allow automatic mixed-precision training
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
    engine.global_trainer(train_dataloader, valid_dataloader, model, optimizer, scheduler, device, nb_epochs, model_path)
    
if __name__ == "__main__":
    args = parser.parse_args()
    run(args.opt_level, args.keep_batchnorm_fp32, args.batch_size,
        args.nb_epochs, args.data_path, args.model_path)