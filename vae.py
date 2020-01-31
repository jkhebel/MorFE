import time
import click
import yaml
from tqdm import tqdm
import logging

from sklearn import metrics

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import HCSData
from models import VAE


logging.basicConfig(level=logging.INFO)

# Parameters
with open("./configs/params.yml", 'r') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)


def vae_loss(output, input, mean, logvar, loss_func):
    recon_loss = loss_func(output, input)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mean**2 - 1. - logvar, 1))
    return recon_loss + kl_loss


# https://becominghuman.ai/variational-autoencoders-for-new-fruits-with-keras-and-pytorch-6d0cfc4eeabd
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD


@click.command()
@click.option("--csv_file", type=click.Path(exists=True), default=p['csv_file'])
@click.option("--data_path", type=click.Path(exists=True), default=p['data_path'])
@click.option("--debug/--no-debug", default=p['debug'])
@click.option("-e", "--epochs", type=int, default=p['epochs'])
@click.option("-b", "--batch_size", type=int, default=p['batch_size'])
@click.option("-B", "--max_batches", type=int, default=p['max_batches'])
@click.option("-s", "--split", type=float, default=p['split'])
@click.option("--parallel/--no-parallel", default=p['parallel'])
@click.option("-f", "--n_features", type=int, default=32)
@click.option("-l", "--n_layers", type=int, default=2)
def train(csv_file, data_path, debug, epochs, batch_size, max_batches, split,
          parallel, n_features, n_layers):

    # If debug, set logger level and log parameters
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(f"CSV File: {csv_file}")
        logging.debug(f"Data Path: {data_path}")
        logging.debug(f"Debug: {debug}")
        logging.debug(f"Epochs: {epochs}")
        logging.debug(f"Batch Size: {batch_size}")
        logging.debug(f"Maximum batches per epoch: {max_batches}")
        logging.debug(f"Test-train split: {split*100}%")
        logging.debug(f"Latent features: {n_features}%")
        logging.debug(f"VAE Layers: {n_layers}%")

    writer = SummaryWriter(
        f'{data_path}/runs/bbbc_{time.strftime("%Y%m%d-%H%M%S")}'
    )

    # Set up gpu/cpu device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Dataset
    data = HCSData.from_csv(csv_file, data_path)  # Load dataset
    data[0][0].shape
    train, test = data.split(split)  # Split data into train and test
    # data[0][0].shape
    train_loader = torch.utils.data.DataLoader(  # Generate a training loader
        train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(  # Generate a testing loader
        test, batch_size=batch_size, shuffle=True)

    in_shape = tuple(data[0][0].shape)
    net = VAE()

    # Move Model to GPU
    if torch.cuda.device_count() > 1:  # If multiple gpu's
        net = torch.nn.DataParallel(net)  # Parallelize
    net.to(device)  # Move model to device

    writer = SummaryWriter(
        f"{data_path}/runs/{net.__class__.__name__}"
        f"_b{batch_size}-{max_batches}_e{epochs}"
        f"_{time.strftime('%Y-%m-%d_%H-%M')}"
    )

    # Define loss and optimizer
    # criterion = torch.nn.MSELoss()
    vae_loss = Loss()
    optimizer = torch.optim.Adam(net.parameters())

    print("Training...")

    # Training
    for epoch in range(epochs):  # Iter through epochcs
        cum_loss = 0
        msg = f"Training epoch {epoch+1}: "
        ttl = max_batches or len(train_loader)  # Iter through batches
        for batch_n, (X, _) in tqdm(enumerate(train_loader), msg, ttl):
            x = X.to(device)  # Move batch samples to gpu

            o, u, logvar = net(x)  # Forward pass
            optimizer.zero_grad()  # Reset gradients
            loss = vae_loss(o, x, u, logvar)  # Compute Loss
            loss.backward()  # Propagate loss, compute gradients
            optimizer.step()  # Update weights

            cum_loss += loss.item()

            in_grid = torchvision.utils.make_grid(
                x.view(5 * batch_size, 1, x.shape[-2], x.shape[-1]),
                nrow=5
            )
            out_grid = torchvision.utils.make_grid(
                o.view(5 * batch_size, 1, o.shape[-2], o.shape[-1]),
                nrow=5
            )

            writer.add_image('Input', in_grid, epoch * max_batches + batch_n)
            writer.add_image('Output', out_grid, epoch * max_batches + batch_n)
            writer.add_scalar(
                'loss',
                loss,
                epoch * max_batches + batch_n
            )

        torch.save(net.state_dict(),
                   f"{data_path}/models/{net.__class__.__name__}"
                   f"_b{batch_size}-{max_batches}_e{epochs}"
                   f"_{time.strftime('%Y-%m-%d_%H-%M')}"
                   )

        print(f"Training loss: {cum_loss:.2f}")


if __name__ == '__main__':
    train()
