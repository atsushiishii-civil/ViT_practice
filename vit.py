import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
### instruction of ViT
## The influence of self-attention in this process is profound 
## because it allows the model to dynamically decide which parts of the image are important
## based on the content of the image itself, 
## rather than relying on fixed geometric kernels as in CNNs. 
## This adaptability can lead to better performance, 
## especially in complex scenes where contextual understanding is crucial.

## 手順
## まずは、画像をグリッドで区切り、パッチで分ける
## Embeddingを通し、各パッチに対して、よりhigh dimensionの情報に。
## それにより各パッチは意味のあるものに。
## また、クラストークンを追加し、分類用のトークンにする。
## さらに、positinal tokenを加え、位置情報を追加する。
## Query, Key, Valueを作成する。
## 各パッチの関連性を計算する。
## どこからどこに対して関連しているかを計算する。
## それによりどのパッチが重要なのかがわかる
## より、"focus"ポイントを抽出して画像分類ができるようになる。
## 最後に、MLPでクラストークンのみを取り出し、画像分類を行う。

class ViT(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 image_size = 256,
                 patch_size = 16,
                 embedding_dim = 384,
                 num_heads = 3,
                 num_layers = 12,
                 hidden_dim = 768,
                 num_classes = 1000,
                 dropout = 0.1):
        super(ViT,self).__init__()
        self.input_layer = ViT_input_Layer(in_channels, image_size, patch_size, embedding_dim, dropout)
        self.encoder_layer = nn.Sequential(*[
            Encoder_Block(num_heads, embedding_dim, dropout)
            for _ in range(num_layers)
        ])
        ## 最後にMLP
        ## (B,D) -> (B,num_classes)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )
        print("ViT is implemented")
    def forward(self, x:torch.Tensor)->torch.Tensor:
        ## (B,C,H,W) -> (B,N+1,D)
        x = self.input_layer(x)
        ## (B,N+1,D) -> (B,N+1,D)
        x = self.encoder_layer(x)
        ## クラストークンのみ取り出す
        x = x[:,0,:]
        ## (B,D) -> (B,1000)
        x = self.mlp_head(x)
        return x

class ViT_input_Layer(nn.Module):
    def __init__(self, in_channels = 3, image_size = 256 , patch_size = 16, embedding_dim = 384,dropout = 0.1):
        super(ViT_input_Layer,self).__init__()
        self.in_channels = in_channels # number of input channels, assuming 3 for RGB images
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size) ** 2 ## number of patches
        self.num_patches_row = image_size // patch_size
        self.flatten_patch_size = patch_size * patch_size * 3 
        ## 1. Patch Embedding
        self.patch_embedding_layer = nn.Conv2d(
            in_channels = self.in_channels,  # number of input channels, 3 for RGB images
            out_channels = self.embedding_dim, # number of output channels, same as embedding dim
            kernel_size = self.patch_size, # kernel size
            stride = self.patch_size) # stride, same as patch size
        ## 2. Adding the CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim)) # CLS token
        ## 3. Adding the positional encoding
        ## The reason why Parameter is used:        
        ## 1. The positional encoding is a learnable parameter, and it is initialized randomly
        ## 2. The positional encoding is added to the input embedding to capture the positional information of the patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embedding_dim)) # positional encoding   
        ## 4. Dropout
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        #########################################################
        ## 1. Patch Embedding ##################################
        #########################################################
        ## x.shape = (batch_size, in_channels, image_size, image_size)
        ## output.shape = (batch_size, embedding_dim, num_patches_row, num_patches_row)
        x = self.patch_embedding_layer(x)
        #########################################################
        ## 2. Flatten the patch embedding #######################
        #########################################################
        ## x.shape = (batch_size, embedding_dim, num_patches_row, num_patches_row)
        ## output.shape = (batch_size, embedding_dim, num_patches)
        x = x.flatten(2)    
        #########################################################
        ## 3. Adding the CLS token ##############################
        #########################################################
        ## x.shape = (batch_size, embedding_dim, num_patches)
        ## output.shape = (batch_size, num_patches, embedding_dim)
        x = x.transpose(1,2)
        #########################################################
        ## 4. Concatenate the CLS token #######################
        #########################################################
        ## x.shape = (batch_size, embedding_dim, num_patches)
        batch_size = x.shape[0]
        ## batch size分のcls_tokenを作成
        # (1, 1, embedding_dim) -> (batch_size, 1, embedding_dim)
        repeat_cls_token = self.cls_token.repeat(batch_size, 1, 1)
        ## x.shape = (batch_size, num_patches, embedding_dim)
        ## repeat_cls_token.shape = (batch_size, 1, embedding_dim)
        ## output.shape = (batch_size, num_patches + 1, embedding_dim)
        x = torch.cat([repeat_cls_token, x], dim=1)

        #########################################################
        ## 5. Adding the positional encoding ##################
        #########################################################
        ## x.shape = (batch_size, num_patches + 1, embedding_dim)
        ## pos_embedding.shape = (1, num_patches + 1, embedding_dim)
        ## output.shape = (batch_size, num_patches + 1, embedding_dim)
        x = x + self.pos_embedding
        #########################################################
        ## 6. Dropout ###########################################
        #########################################################
        ## x.shape = (batch_size, num_patches + 1, embedding_dim)
        ## output.shape = (batch_size, num_patches + 1, embedding_dim)
        x = self.dropout_layer(x)
        return x ## output.shape = (batch_size, num_patches + 1, embedding_dim)


class Self_Attention_Layer(nn.Module):
    ## this layer is responsible for the self-attention mechanism
    ## it employs multi-head attention, which is a technique to allow the model to attend to different representation subspaces
    ## in the self-attention mechanism, the model learns to attend to different representation subspaces
    def __init__(self, num_heads = 3, embedding_dim = 384):
        super(Self_Attention_Layer,self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.sqrt_dh = self.head_dim**0.5
        # 1. Create the query, key, and value linear layers
        self.W_q = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        # 2. Create the softmax layer
        self.softmax = torch.nn.Softmax(dim=-1)
        # 3. Attention drop layer
        self.attn_drop = nn.Dropout(0)
        # 3. Create the output linear layer
        self.output_linear = torch.nn.Linear(embedding_dim, embedding_dim)
    def forward(self, z):
        ## z.shape = (batch_size, num_patches + 1, embedding_dim) = (2,5,384)
        batch_size, num_patch,_ = z.shape
        num_heads = self.num_heads # num_heads = 3
        ###################################################################
        ## 1. Embedding ###################################################
        ###################################################################
        ## query, Key, Valueを抽出
        q = self.W_q(z) ## (batch_size, num_patches + 1, embedding_dim)
        k = self.W_k(z) ## (batch_size, num_patches + 1, embedding_dim)
        v = self.W_v(z) ## (batch_size, num_patches + 1, embedding_dim)
        ###################################################################
        ## 2. Split heads ##################################################
        ###################################################################
        ## (B,N+1,D) -> (B,N+1,h,D/h) 
        ## B : batch_size
        ## N : num_patch
        ## D : embedding_dim
        ## h : num_heads
        ## .view()によって、うまくサイズを調整
        q = q.view(batch_size, num_patch, num_heads, self.head_dim)
        k = k.view(batch_size, num_patch, num_heads, self.head_dim)
        v = v.view(batch_size, num_patch, num_heads, self.head_dim)
        ## (B,N+1,h,D/h) -> (B,h,N+1,D/h)
        ## 入れ替え
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        ###################################################################
        ## 3. Scaled Dot Product Attention #################################
        ###################################################################
        ## (B,h,N+1,D/h) -> (B,h,D/h,N+1)
        k_T = k.transpose(2, 3)
        ## 内積計算 (K,Q) 
        ## Q : (B,h,N+1,D/h)
        ## K_T : (B,h,D/h,N+1)    
        ## (B,h,N+1,D/h)・(B,h,D/h,N+1) -> (B,h,N+1,N+1)
        ## multi head がないと (B,N+1,N+1)になってしまう。
        ## より多くの情報を得るためhead数を増やす
        dot_k_q = (q@k_T)/self.sqrt_dh
        ## 行方向にsoftmaxを適用 
        attn = F.softmax(dot_k_q, dim=-1)
        ###################################################################
        ## 4. Valueの重み付け ################################################
        ###################################################################
        ## attn : (B,h,N+1,N+1)
        ## v : (B,h,N+1,D/h)
        ## (B,h,N+1,N+1)・(B,h,N+1,D/h) -> (B,h,N+1,D/h)
        scaled_attention = attn@v
        ## Transpose the output
        ## (B,h,N+1,N+1)
        scaled_attention = scaled_attention.transpose(1, 2)
        ## 合体させる
        ## (B,N+1,h,D/h) -> (B,N+1,D)
        original_size_attention = scaled_attention.reshape(batch_size, num_patch, self.embedding_dim)
        ###################################################################
        ## 5. Output Linear ################################################
        ###################################################################
        ## original_size_attention.shape = (B,N+1,D)
        ## output.shape = (B,N+1,D) 
        output = self.output_linear(original_size_attention)
        return output

class Encoder_Block(nn.Module):
    ## Encode Block consists of the following components
    ## 1. LayerNorm
    ## 2. Multi-head attention
    ## 3. Addition (Skip connection)
    ## 4. LayerNorm
    ## 5. MLP
    ## 6. Addition (Skip connection)
    def __init__(self, num_heads = 3, embedding_dim = 384,dropout = 0.1):
        super(Encoder_Block,self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.hidden_dim = embedding_dim * 4
        ###################################################################
        ## 1. LayerNorm ###################################################
        ###################################################################
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        ###################################################################
        ## 2. Multi-head attention ########################################
        ###################################################################
        self.multi_head_attention = Self_Attention_Layer(num_heads, embedding_dim)
        ###################################################################
        ## 3. Second LayerNorm ##############################################
        ###################################################################
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        ## 4. MLP ##############################################################
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z) -> torch.Tensor:
        ## z.shape = (batch_size, num_patches + 1, embedding_dim)
        ## output.shape = (batch_size, num_patches + 1, embedding_dim)
        ###################################################################
        ## 1. LayerNorm ###################################################
        ###################################################################
        z1 = self.layer_norm_1(z)
        ###################################################################
        ## 2. Multi-head attention and skip connection ####################
        ###################################################################
        z2 = self.multi_head_attention(z1) + z
        ###################################################################
        ## 3. Second LayerNorm ############################################
        ###################################################################
        z3 = self.layer_norm_2(z2)
        ###################################################################
        ## 4. MLP and skip connection #####################################    
        ###################################################################
        out = self.mlp(z3) + z2
        ## output.shape = (batch_size, num_patches + 1, embedding_dim)
        return out 

