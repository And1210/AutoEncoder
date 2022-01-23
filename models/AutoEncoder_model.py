import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class Encoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=24,
            out_channels=48,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.bn = nn.BatchNorm2d(num_features=48)
        self.relu = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten(start_dim=1)

        self.lin1 = nn.Linear(254016, 128)
        self.lin2 = nn.Linear(128, encoded_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.lin1 = nn.Linear(encoded_dim, 128)
        self.lin2 = nn.Linear(128, 254016)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 63, 63))

        self.deconv1 = nn.ConvTranspose2d(64, 48, 3, stride=2, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(48, 24, 3, stride=2, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(24, 3, 3, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(24)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)

        x = self.unflatten(x)

        x = self.deconv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)

        return x

#PyTorch defined model
class AutoEncoder(nn.Module):
    """basenet for fer2013"""
    def __init__(self, encoded_dim=256):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(encoded_dim)
        self.decoder = Decoder(encoded_dim)

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x

#The abstract model class, uses above defined class and is used in the train script
class AutoEncodermodel(BaseModel):
    """basenet for fer2013"""

    def __init__(self, configuration):
        super().__init__(configuration)

        #Initialize model defined above
        self.model = AutoEncoder(configuration['encoded_dim'])
        self.model.cuda()

        #Define loss function
        self.criterion_loss = nn.MSELoss().cuda()
        #Define optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=configuration['lr'],
            betas=(configuration['momentum'], 0.999),
            weight_decay=configuration['weight_decay']
        )

        #Need to include these arrays with the optimizers and names of loss functions and models
        #Will be used by other functions for saving/loading
        self.optimizers = [self.optimizer]
        self.loss_names = ['total']
        self.network_names = ['model']

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

    #Calls the models forwards function
    def forward(self):
        x = self.input
        self.output = self.model.forward(x)
        return self.output

    #Computes the loss with the specified name (in this case 'total')
    def compute_loss(self):
        # print(self.output.shape)
        # print(self.label.shape)
        self.loss_total = self.criterion_loss(self.output, self.label)

    #Compute backpropogation for the model
    def optimize_parameters(self):
        self.loss_total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

    #Test function for the model
    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        self.val_images.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)

    #Should be run after each epoch, outputs accuracy
    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['Accuracy'] = val_accuracy

        if (visualizer != None):
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []


if __name__ == "__main__":
    net = TEMPLATEmodel().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
