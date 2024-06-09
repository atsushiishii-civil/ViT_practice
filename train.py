import torch
import torch.nn as nn
from vit import ViT
import matplotlib.pyplot as plt
import time
import os
import tqdm
import pandas as pd
import glob

def train(net,
          train_dataloader,
          val_dataloader,
          test_dataloader,
          num_epochs,
          optimizer,
          criterion,
          device):
    print("Device we use:", device)
    ## 1. Move the model to the device
    net.to(device)
    ## 2. Train the model 
    train_loss_list = []
    val_loss_list = []
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    net, optimizer, start_epoch = load_checkpoint(net, optimizer, 'weights/')
    for epoch in range(start_epoch + 1, num_epochs+1):
        # 開始時刻を保存
        t_epoch_start = time.time()
        print('-------------')
        print('Epoch {}/{} starts'.format(epoch, num_epochs))
        train_loss = 0.0
        for x, y in tqdm.tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            ## 2.1. Forward pass
            pred = net(x)
            ## 2.2. Backward pass
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del loss  # メモリ節約のため
        train_loss_list.append(train_loss)
        ## Print the loss
        print(f"Epoch [{epoch}/{num_epochs}], Train loss: {train_loss}")
        val_loss = 0.0
        for x, y in tqdm.tqdm(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            loss = criterion(pred, y)
            val_loss += loss.item()
            del loss  # メモリ節約のため
        val_loss_list.append(val_loss)
        print(f"Epoch [{epoch}/{num_epochs}], Val loss: {val_loss}")
        t_epoch_finish = time.time()
        print(f"Epoch {epoch}/{num_epochs} Ends, Time: {t_epoch_finish - t_epoch_start}")
        t_epoch_start = time.time()
        save_checkpoint(net, optimizer, epoch, filename = 'weights/ViT_' +
                       str(epoch) + '.pth')
        ## 3. Save the loss
        if os.path.exists("log_output.csv"):
            df = pd.read_csv("log_output.csv")
            df = pd.concat([df, pd.DataFrame({"epoch": [epoch], "train_loss": [train_loss], "val_loss": [val_loss]})], ignore_index=True)
        else:
            df = pd.DataFrame([])
            df["epoch"] = [epoch]
            df["train_loss"] = [train_loss]
            df["val_loss"] = [val_loss]
        df.to_csv("log_output.csv", index=False)
    ## 4. Plot the loss
    df = pd.read_csv("log_output.csv")
    train_loss_list = df["train_loss"].tolist()
    val_loss_list = df["val_loss"].tolist()
    plt.plot(df["epoch"].tolist(), train_loss_list, label="Train loss")
    plt.plot(df["epoch"].tolist(), val_loss_list, label="Val loss")
    plt.legend()
    plt.show()

def save_checkpoint(net, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

def load_checkpoint(net, optimizer, folder):
    files = glob.glob(folder + "/*.pth")
    ## If there is no file in the folder, return 0
    if len(files) == 0:
        return net, optimizer, 0
    ## Sort the files by epoch number
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    ## Get the last file
    filename = files[-1]
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from {filename}")
    return net, optimizer, epoch

