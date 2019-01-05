##########################################
# Stat231 Project 1:
# Autoencoder
# Author:
##########################################

import torch
import torch.nn as nn
import torch.optim as optim


class appearance_autoencoder(nn.Module):
    def __init__(self):
        super(appearance_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # TODO: Fill in the encoder structure
        )

        self.fc1 = nn.Sequential(                    
            # TODO: Fill in the FC layer structure
        )

        self.decoder = nn.Sequential(
            # TODO: Fill in the decoder structure
            # Hint: De-Conv in PyTorch: ConvTranspose2d 
        )

    def forward(self, x):
            # TODO: Fill in forward pass
        return x_recon


class landmark_autoencoder(nn.Module):
    def __init__(self):
        super(landmark_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # TODO: Fill in the encoder structure
        )
        self.decoder = nn.Sequential(
            # TODO: Fill in the decoder structure
        )

    def forward(self, x):
            # TODO: Fill in forward pass
        return x_recon


class autoencoder(object):
    def __init__(self, appear_lr, landmark_lr, use_cuda):
        self.appear_model = appearance_autoencoder()
        self.landmark_model = landmark_autoencoder()
        self.use_cuda = use_cuda
        if use_cuda:
            self.appear_model.cuda()
            self.landmark_model.cuda()
        self.criterion = nn.MSELoss()
        self.appear_optim = optim.Adam(self.appear_model.parameters(), lr=appear_lr)
        self.landmark_optim = optim.Adam(self.landmark_model.parameters(), lr=landmark_lr)
        
    def train_appear_model(self, epochs, trainloader):
        self.appear_model.train()
        epoch = 0
        # TODO: Train appearance autoencoder

    def train_landmark_model(self, epochs, trainloader):
        self.landmark_model.train()
        epoch = 0
        # TODO: Train landmark autoencoder

    def test_appear_model(self, testloader):
        self.appear_model.eval()
        # TODO: Test appearance autoencoder
    
    def test_landmark_model(self, testloader):
        self.landmark_model.eval()
        # TODO: Test landmark autoencoder

