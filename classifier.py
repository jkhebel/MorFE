import time
from tqdm import tqdm
import click
import yaml
import logging

import numpy as np
from sklearn import metrics

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import HCSData
import models

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


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
    try:
        logging.debug(f"Device: {torch.cuda.get_device_name(0)}")
    except AssertionError as e:
        logging.debug(e)

    # Dataset
    data = HCSData.from_csv(csv_file, data_path)  # Load dataset
    train, test = data.split(0.8)  # Split data into train and test
    logging.debug(f"Data loaded and split")

    # Sampler (random weighted)
    class_weights = train.class_weights
    sampler = torch.utils.data.WeightedRandomSampler(
        class_weights, max_batches * batch_size or len(class_weights),
        replacement=False)

    train_loader = torch.utils.data.DataLoader(  # Generate a training data loader
        train, batch_size=batch_size, sampler=sampler
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

    writer = SummaryWriter(
        f"{data_path}/runs/{net.__class__.__name__}"
        f"_b{batch_size}-{max_batches}_e{epochs}"
        f"_{time.strftime('%Y-%m-%d_%H-%M')}"
    )

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print("\n\nTraining...\n")

    try:
        # Training
        for epoch in range(epochs):  # Iter through epochcs
            cum_loss = 0
            tr_predictions = []
            tr_labels = []
            msg = f"Training epoch {epoch+1}: "
            for batch_n, (X, Y) in tqdm(enumerate(train_loader), msg, max_batches or len(train_loader)):
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

                grid = torchvision.utils.make_grid(
                    x.view(5 * batch_size, 1, x.shape[-2], x.shape[-1]),
                    nrow=5
                )

                writer.add_image('Image_batch', grid, epoch *
                                 max_batches + batch_n)
                writer.add_scalar(
                    'tr_accuracy',
                    metrics.accuracy_score(y.cpu(), predicted.cpu()),
                    epoch * max_batches + batch_n
                )
                writer.add_scalar(
                    'tr_F1',
                    metrics.f1_score(y.cpu(), predicted.cpu(),
                                     zero_division=0),
                    epoch * max_batches + batch_n
                )
                writer.add_scalar(
                    'tr_precision',
                    metrics.precision_score(
                        y.cpu(), predicted.cpu(), zero_division=0),
                    epoch * max_batches + batch_n
                )
                writer.add_scalar(
                    'tr_recall',
                    metrics.recall_score(
                        y.cpu(), predicted.cpu(), zero_division=0),
                    epoch * max_batches + batch_n
                )
                try:
                    writer.add_scalar(
                        'tr_auroc',
                        metrics.roc_auc_score(y.cpu(), predicted.cpu()),
                        epoch * max_batches + batch_n
                    )
                except:
                    logging.debug("Couldn't write auroc to board")
                writer.add_scalar(
                    'loss',
                    loss,
                    epoch * max_batches + batch_n
                )

            tr_predictions = torch.cat(tr_predictions, dim=0).cpu()
            tr_labels = torch.cat(tr_labels, dim=0).cpu()
            print(f"Training loss: {cum_loss/max_batches:.2f}")

            torch.save(net.state_dict(),
                       f"{data_path}/models/{net.__class__.__name__}"
                       f"_b{batch_size}-{max_batches}_e{epochs}"
                       f"_{time.strftime('%Y-%m-%d_%H-%M')}"
                       )

            with torch.no_grad():

                ts_predictions = []
                ts_labels = []
                # Iter through batches
                msg = f"Testing epoch {epoch+1}: "
                ttl = np.ceil((1 - split) * max_batches) or len(test_loader)
                for batch_n, (X, Y) in tqdm(enumerate(test_loader), msg, ttl):
                    # Move batch samples to gpu
                    x, y = X.to(device), Y.to(device)
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
                print(
                    f'Accuracy of the network on the test images: {acc:0.2f}')

                tr_F1 = metrics.f1_score(
                    tr_labels, tr_predictions, zero_division=0)
                F1 = metrics.f1_score(
                    ts_labels, ts_predictions, zero_division=0)
                print(f'F1 of the network on the train images: {tr_F1:0.2f}')
                print(f'F1 of the network on the test images: {F1:0.2f}')

                tr_auroc = metrics.roc_auc_score(tr_labels, tr_predictions)
                auroc = metrics.roc_auc_score(ts_labels, ts_predictions)
                print(
                    f'AUROC of the network on the train images: {tr_auroc:0.2f}')
                print(f'AUROC of the network on the test images: {auroc:0.2f}')

                try:
                    writer.add_image('Image_batch', grid)
                    writer.add_scalar(
                        'accuracy',
                        acc,
                        epoch * ttl + batch_n
                    )
                    writer.add_scalar(
                        'F1',
                        F1,
                        epoch * ttl + batch_n
                    )
                    writer.add_scalar(
                        'auroc',
                        auroc,
                        epoch * ttl + batch_n
                    )
                except:
                    pass

                scheduler.step(F1)

    except (KeyboardInterrupt, SystemExit):
        torch.save(net.state_dict(),
                   f"{data_path}/models/{net.__class__.__name__}"
                   f"_b{batch_size}-{max_batches}_e{epochs}"
                   f"_{time.strftime('%Y-%m-%d_%H-%M')}"
                   )


if __name__ == '__main__':
    train()
