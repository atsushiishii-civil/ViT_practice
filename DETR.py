import torch
import torch.nn as nn
import torchvision.models as models

class DETR(nn.Module):
    def __init__(self, num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()
        # Load a pre-trained ResNet-50 model and remove the classification head
        self.backbone = models.resnet50(pretrained=True)
        del self.backbone.fc
        
        # Convolution to reduce the feature space dimensionality
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        
        # Positional encoding for the transformer
        self.positional_encoding = nn.Parameter(torch.randn(100, hidden_dim))
        
        # Output linear layers
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        
        # Activation and final processing
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        # Get the features from the backbone
        features = self.backbone(inputs)
        
        # Reduce the dimensionality
        features = self.conv(features)
        
        # Flatten the features and add positional encoding
        src = features.flatten(2).permute(2, 0, 1)
        src += self.positional_encoding[:src.size(0)]
        
        # Transformer
        hs = self.transformer(src, src)
        
        # Output heads
        outputs_class = self.linear_class(hs)
        outputs_bbox = self.linear_bbox(hs)
        
        # Activation for bounding box coordinates
        outputs_bbox = self.activation(outputs_bbox)
        
        return outputs_class, outputs_bbox


    def train(self, train_loader, optimizer, criterion, num_epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        # Link optimizer to the model's parameters
        self.train() # Set the model to training mode
        for epoch in range(num_epochs):
            for images, targets in train_loader:
                optimizer.zero_grad()
                outputs_class, outputs_bbox = self.forward(images)
                
                # Calculate loss
                loss_class = criterion(outputs_class, targets['labels'])
                loss_bbox = criterion(outputs_bbox, targets['boxes'])
                loss = loss_class + loss_bbox
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



