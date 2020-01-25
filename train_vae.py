import click
import yaml
from tqdm import tqdm
import logging

import torch

from dataset import HCSData
from models import VAE


logging.basicConfig(level=logging.INFO)

# Parameters
with open("./configs/params.yml", 'r') as f:
    p = yaml.load(f)


def vae_loss(output, input, mean, logvar, loss_func):
    recon_loss = loss_func(output, input)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mean**2 - 1. - logvar, 1))
    return recon_loss + kl_loss


@click.command()
@click.option("--csv_file", type=click.Path(exists=True), default=p['csv_file'])
@click.option("--debug/--no-debug", default=p['debug'])
@click.option("-e", "--epochs", type=int, default=p['epochs'])
@click.option("-b", "--batch_size", type=int, default=p['batch_size'])
@click.option("-s", "--split", type=float, default=p['split'])
@click.option("-z", "--n_features", type=int, default=32)
def train(csv_file, debug, epochs, batch_size, split, n_features):
    # Set up gpu/cpu device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset
    data = HCSData.from_csv('data/mini.csv')  # Load dataset
    train, test = data.split(0.8)  # Split data into train and test

    train_loader = torch.utils.data.DataLoader(  # Generate a training data loader
        train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(  # Generate a testing data loader
        test, batch_size=batch_size, shuffle=False)

    in_shape = tuple(data[0][0].shape)
    net = VAE(in_shape, n_features)

    # Move Model to GPU
    if torch.cuda.device_count() > 1:  # If multiple gpu's
        net = torch.nn.DataParallel(net)  # Parallelize
    net.to(device)  # Move model to device

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print("Training...")

    # Training
    for epoch in range(epochs):  # Iter through epochcs
        cum_loss = 0
        msg = f"Training epoch {epoch+1}: "
        ttl = len(train_loader)  # Iter through batches
        for batch_n, (X, _) in tqdm(enumerate(train_loader), msg, ttl):
            x = X.to(device)  # Move batch samples to gpu

            o, u, logvar = net(x)  # Forward pass
            optimizer.zero_grad()  # Reset gradients
            # loss = criterion(o, x)
            loss = vae_loss(o, x, u, logvar, criterion)  # Compute Loss
            loss.backward()  # Propagate loss, compute gradients
            optimizer.step()  # Update weights

            print(loss.item())

            cum_loss += loss.item()

            # tqdm.write((
            #     f"Batch {batch_n+1}:"
            #     f"\tLoss: {loss.item():.4f}"
            #     f"\tPrediction: {o.argmax()}"
            #     f" \t Label: {y.item()}"
            # ))

        # scheduler.step(metric)  # Update the learning rate

        print(f"Training loss: {cum_loss:.2f}")


if __name__ == '__main__':
    train()
