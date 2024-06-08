import torch
import torch.nn as nn
from vit import ViT
import matplotlib.pyplot as plt
import time
import os
import tqdm
import pandas as pd

def train(net,
          train_dataloader,
          val_dataloader,
          test_dataloader,
          epochs,
          optimizer,
          criterion,
          device):
    print("Device we use:", device)
    ## 1. Move the model to the device
    net.to(device)
    ## 2. Train the model 
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        print('-------------')
        print('Epoch {}/{} starts'.format(epoch+1, epochs))
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
        print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss}")
        val_loss = 0.0
        for x, y in tqdm.tqdm(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            loss = criterion(pred, y)
            val_loss += loss.item()
            del loss  # メモリ節約のため
        val_loss_list.append(val_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Val loss: {val_loss}")
        t_epoch_finish = time.time()
        print(f"Epoch [{epoch + 1}/{epochs}] Ends, Time: {t_epoch_finish - t_epoch_start}")
        t_epoch_start = time.time()
        # ネットワークを保存する
        if not os.path.exists('weights/'):
            os.makedirs('weights/')
        if ((epoch+1) % 50 == 0):
            torch.save(net.state_dict(), 'weights/ViT_' +
                       str(epoch+1) + '.pth')
    ## 3. Save the loss
    df = pd.DataFrame([])
    df["epoch"] = [i for i in range(1, epochs+1)]
    df["train_loss"] = train_loss_list
    df["val_loss"] = val_loss_list
    df.to_csv("log_output.csv")
    ## 4. Plot the loss
    plt.plot(train_loss_list, label="Train loss")
    plt.plot(val_loss_list, label="Val loss")
    plt.legend()
    plt.show()

