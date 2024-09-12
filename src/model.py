import torch.nn as nn

class CRNN(nn.Module):  
    def __init__(self, num_classes, channels):
        super(CRNN, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 0), stride=(1, 2)),


            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 0), stride=(1, 2)),


            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), padding=(0, 0), stride=(2, 2)),


            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), padding=(0, 0), stride=(2, 2)),


            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), padding=(0, 0), stride=(2, 2)),

            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 0), stride=(1, 2)),
        )   

        self.rnns = nn.Sequential(
            nn.LSTM(1024, 512, batch_first=True, bidirectional=True),
            nn.LSTM(1024, 512, batch_first=True, bidirectional=True),
        )
        
        self.classifier = nn.Sequential(
           nn.Linear(1024, num_classes),
        )
        

    def forward(self, x):
        x = self.convs(x)
        
        x = x.squeeze(3) 
        x = x.permute(0, 2, 1)  

        for lstm in self.rnns:
            x, _ = lstm(x)

        x = self.classifier(x)

        x = x.log_softmax(dim=2)
        return x