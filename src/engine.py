import tqdm
import sys
import time
import torch
from apex import amp
import torch.nn as nn
from sklearn.metrics import accuracy_score



def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_step(ids, mask, token_type_ids, targets, model, phase, optimizer, scheduler):
    # Put the gradients to zero
    model.zero_grad()
    # Compute the outputs
    outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids    
        )
    # Compute the loss   
    loss = loss_fn(outputs, targets)
    batch_loss = loss.item()
    
    # Compute the accuracy
    targets = targets.cpu().detach().numpy()
    outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
    score = accuracy_score(targets, outputs)
    
    # Compute the gradient in train mode   
    if phase == "train": 
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            #loss.backward()
        optimizer.step()
        scheduler.step()
    
    return batch_loss, score

            
def global_trainer(train_dataloader, valid_dataloader, model, optimizer, scheduler, device, nb_epochs):
    
    # For each epoch
    for epoch in range(nb_epochs):
        # We train or valid for the whole dataset
        with tqdm.tqdm(total=len(train_dataloader), file=sys.stdout, leave=True, desc='Epoch ', \
                       bar_format="{l_bar}{bar:20}{r_bar}{bar:-15b}") as pbar:
            
        #
            best_accuracy = 0
        
            for phase in ["train", "valid"]:
                total_loss = 0
                total_acc_score = 0
                
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                for batch, d in enumerate(train_dataloader):
                    pbar.set_description('Epoch {:>3}'.format(epoch + 1))
      
                    # Take the bacth data
                    ids = d["ids"].to(device, dtype=torch.long)
                    token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
                    mask = d["mask"].to(device, dtype=torch.long)
                    targets = d["targets"].to(device, dtype=torch.float)
                
                    if phase == "train":
                        batch_loss, score = train_step(ids,
                                                       mask,
                                                       token_type_ids,
                                                       targets,
                                                       model,
                                                       phase,
                                                       optimizer,
                                                       scheduler)
                    else:
                        with torch.no_grad(): # to save some gpu memory
                            batch_loss, score = train_step(ids,
                                                           mask, 
                                                           token_type_ids, 
                                                           targets,
                                                           model,
                                                           phase,
                                                           optimizer,
                                                           scheduler)
          
                        
                    total_loss += batch_loss
                    total_acc_score += score
              
                
          
                    if phase == 'train':
                        train_loss = total_loss / (batch + 1)
                        train_acc = total_acc_score / (batch + 1)
                        pbar.set_postfix_str('Train loss {:.4f} Train Acc {:.4f}'.format(train_loss, train_acc))
                        pbar.update(1)
                        time.sleep(1)
                        
                    else:
                        val_loss = total_loss / (batch + 1)
                        val_acc = total_acc_score / (batch + 1)
                        pbar.set_postfix_str('Train loss {:.4f} Train Acc {:.4f} Val loss {:.4f} Val Acc {:.4f}' \
                                             .format(train_loss, train_acc, val_loss, val_acc))
                        time.sleep(1)  
                    # Save the checkpoint with automatic mixed precision at the end of the epoch 
                    if phase == "valid":
                        if val_acc > best_accuracy:
                            best_accuracy = val_acc
                            checkpoint = {'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'amp': amp.state_dict()
                                          }
                            torch.save(checkpoint, 'amp_checkpoint.pt')
         