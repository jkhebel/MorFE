import time
import click
import yaml
from tqdm import tqdm
import logging

import numpy as np

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import HCSData
from models import VAE

from skimage.metrics import mean_squared_error as mse


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
@click.option("-bf", "--n_base_features", type=int, default=32)
@click.option("-lf", "--n_latent_features", type=int, default=32)
@click.option("-l", "--n_layers", type=int, default=2)
def train(csv_file, data_path, debug, epochs, batch_size, max_batches, split,
          parallel, n_base_features, n_latent_features, n_layers):

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
        logging.debug(f"Base features: {n_base_features}")
        logging.debug(f"Latent features: {n_latent_features}")
        logging.debug(f"VAE Layers: {n_layers}")

    # Set up gpu/cpu device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Dataset
    data = HCSData.from_csv(csv_file, data_path)  # Load dataset
    logging.debug('Data loaded')
    train, test = data.split(split)  # Split data into train and test
    # data[0][0].shape
    train_loader = torch.utils.data.DataLoader(  # Generate a training loader
        train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(  # Generate a testing loader
        test, batch_size=batch_size, shuffle=True)

    net = VAE(n_layers=6, lf=n_latent_features, base=n_base_features)
    logging.debug(net)

    # Move Model to GPU
    if torch.cuda.device_count() > 1:  # If multiple gpu's
        net = torch.nn.DataParallel(net)  # Parallelize
    net.to(device)  # Move model to device

    tr_writer = SummaryWriter(
        f"{data_path}/runs/training_{time.strftime('%Y-%m-%d_%H-%M')}")
    vl_writer = SummaryWriter(
        f"{data_path}/runs/validation_{time.strftime('%Y-%m-%d_%H-%M')}")

    # Define loss and optimizer
    # criterion = torch.nn.MSELoss()
    vae_loss = Loss()
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    print("Training...")

    try:
        # Training
        for epoch in range(epochs):  # Iter through epochcs
            cum_loss = 0
            msg = f"Training epoch {epoch+1}: "
            ttl = max_batches or len(train_loader)  # Iter through batches
            for batch_n, (X, _) in tqdm(enumerate(train_loader), msg, ttl):

                if batch_n > max_batches:
                    break

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

                if batch_n % 8 == 0:
                    tr_writer.add_image('Input', in_grid, epoch *
                                        max_batches + batch_n)
                    tr_writer.add_image('Output', out_grid, epoch *
                                        max_batches + batch_n)
                    # writer.add_image('Mean', u_grid, epoch *
                    #                  max_batches + batch_n)
                    # writer.add_image('Logvar', logvar_grid, epoch *
                    #                  max_batches + batch_n)

                tr_writer.add_scalar(
                    'loss',
                    loss,
                    epoch * max_batches + batch_n
                )
                tr_writer.add_scalar(
                    'mse',
                    mse(x.cpu().detach().numpy(), o.cpu().detach().numpy()),
                    epoch * max_batches + batch_n
                )

            with torch.no_grad():

                val_loss = 0
                val_mse = []
                # Iter through batches
                msg = f"Testing epoch {epoch+1}: "
                ttl = max_batches or len(test_loader)
                for batch_n, (X, _) in tqdm(enumerate(test_loader), msg, ttl):

                    if batch_n > max_batches:
                        break

                    # Move batch samples to gpu
                    x = X.to(device)
                    o, u, logvar = net(x)  # Forward pass

                    loss = vae_loss(o, x, u, logvar)
                    val_loss += loss.item()

                    val_mse.append(mse(x.cpu().detach().numpy(),
                                       o.cpu().detach().numpy()))

                vl_writer.add_scalar(
                    'loss',
                    val_loss,
                    epoch * max_batches + batch_n
                )
                vl_writer.add_scalar(
                    'mse',
                    np.mean(val_mse),
                    epoch * max_batches + batch_n
                )

            scheduler.step(-cum_loss)

            torch.save(net.state_dict(),
                       f"{data_path}/models/{net.__class__.__name__}"
                       f"_base-{n_base_features}_latent-{n_latent_features}"
                       f"_{time.strftime('%Y-%m-%d_%H-%M')}.pt"
                       )

    except (KeyboardInterrupt, SystemExit):
        print("Saving model...")
        torch.save(net.state_dict(),
                   f"{data_path}/models/{net.__class__.__name__}"
                   f"_base-{n_base_features}_latent-{n_latent_features}"
                   f"_{time.strftime('%Y-%m-%d_%H-%M')}.pt"
                   )
        print("Model saved.")


if __name__ == '__main__':
    train()
