from tqdm import tqdm
import click
import yaml
import logging

# import cli_args
import numpy as np
from sklearn import metrics
import torch

from dataset import HCSData
import models


# Parameters
with open("./configs/params.yml", 'r') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)


@click.command()
@click.option("-n", "--network", type=str, default=p['network'])
@click.option("--csv_file", type=click.Path(exists=True), default=p['csv_file'])
@click.option("--data_path", type=click.Path(exists=True), default=p['data_path'])
@click.option("--debug/--no-debug", default=p['debug'])
@click.option("-e", "--epochs", type=int, default=p['epochs'])
@click.option("-b", "--batch_size", type=int, default=p['batch_size'])
@click.option("-B", "--max_batches", type=int, default=p['max_batches'])
@click.option("-s", "--split", type=float, default=p['split'])
@click.option("--parallel/--no-parallel", default=p['parallel'])
def train(network, csv_file, data_path, debug, epochs,
          batch_size, max_batches, split, parallel):

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

    # Set up gpu/cpu device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.debug(f"Device: {torch.cuda.get_device_name(0)}")

    # Dataset
    data = HCSData.from_csv(csv_file, data_path)  # Load dataset
    train, test = data.split(0.8)  # Split data into train and test
    logging.debug(f"Data loaded and split")

    train_loader = torch.utils.data.DataLoader(  # Generate a training data loader
        train, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(  # Generate a testing data loader
        test, batch_size=batch_size, shuffle=True)
    logging.debug(f"Data loaders generated")

    # Define Model
    net = getattr(models, network.upper())()
    logging.debug(f"Model built:")
    logging.debug(net)

    # Move Model to GPU
    if (torch.cuda.device_count() > 1) & parallel:  # If multiple gpu's
        net = torch.nn.DataParallel(net)  # Parallelize
        logging.debug(f"Parallelized to {torch.cuda.device_count()} GPUs")
    net.to(device)  # Move model to device

    # Define loss and optimizer
    class_weights = data.class_weights.to(device)
    logging.debug(f"Class weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print("\n\nTraining...\n")

    # Training
    for epoch in range(epochs):  # Iter through epochcs
        cum_loss = 0
        tr_predictions = []
        tr_labels = []
        msg = f"Training epoch {epoch+1}: "
        ttl = max_batches or len(train_loader)  # Iter through batches
        for batch_n, (X, Y) in tqdm(enumerate(train_loader), msg, ttl):
            x, y = X.to(device), Y.to(device)  # Move batch samples to gpu

            o = net(x)  # Forward pass
            optimizer.zero_grad()  # Reset gradients
            loss = criterion(o, y)  # Compute Loss
            loss.backward()  # Propagate loss, compute gradients
            optimizer.step()  # Update weights

            cum_loss += loss.item()

            _, predicted = torch.max(o.data, 1)
            tr_predictions.append(predicted)
            tr_labels.append(y)

            if (batch_n > ttl):
                break

        tr_predictions = torch.cat(tr_predictions, dim=0).cpu()
        tr_labels = torch.cat(tr_labels, dim=0).cpu()
        print(f"Training loss: {cum_loss/ttl:.2f}")

        with torch.no_grad():

            ts_predictions = []
            ts_labels = []
            # Iter through batches
            msg = f"Testing epoch {epoch+1}: "
            ttl = np.ceil((1 - split) * max_batches) or len(test_loader)
            for batch_n, (X, Y) in tqdm(enumerate(test_loader), msg, ttl):
                x, y = X.to(device), Y.to(device)  # Move batch samples to gpu
                o = net(x)  # Forward pass

                _, predicted = torch.max(o.data, 1)
                ts_predictions.append(predicted)
                ts_labels.append(y)

                if debug:
                    tqdm.write(f"\ny : o\n-----")
                    for i in range(len(y)):
                        tqdm.write(f"{y[i]} : {predicted[i]}")

                if (batch_n > ttl):
                    break

            ts_predictions = torch.cat(ts_predictions, dim=0).cpu()
            ts_labels = torch.cat(ts_labels, dim=0).cpu()

            print(f"Epoch {epoch}:")
            # Metrics
            tr_acc = metrics.accuracy_score(tr_labels, tr_predictions)
            acc = metrics.accuracy_score(ts_labels, ts_predictions)
            print(
                f'Accuracy of the network on the train images: {tr_acc:0.2f}')
            print(f'Accuracy of the network on the test images: {acc:0.2f}')

            tr_F1 = metrics.f1_score(
                tr_labels, tr_predictions, zero_division=0)
            F1 = metrics.f1_score(ts_labels, ts_predictions, zero_division=0)
            print(f'F1 of the network on the train images: {tr_F1:0.2f}')
            print(f'F1 of the network on the test images: {F1:0.2f}')

            tr_auroc = metrics.roc_auc_score(tr_labels, tr_predictions)
            auroc = metrics.roc_auc_score(ts_labels, ts_predictions)
            print(f'AUROC of the network on the train images: {tr_auroc:0.2f}')
            print(f'AUROC of the network on the test images: {auroc:0.2f}')

            scheduler.step(F1)


if __name__ == '__main__':
    train()
