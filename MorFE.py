import click
import logging
import yaml
import time

from tqdm import tqdm

import numpy as np
import torch
import torchvision

from skimage.metrics import mean_squared_error as mse

import matplotlib
matplotlib.use("Agg")
from dataset import HCSData
from models import VAE_fm


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option("--dataset", type=click.Path(exists=True), required=True)
@click.pass_context
def cli(ctx, debug, dataset):

  with open("./configs/default_params.yml", 'r') as f:
    ctx = yaml.load(f, Loader=yaml.FullLoader)

  ctx.ensure_object(dict)
  ctx['dataset'] = dataset

  logging.basicConfig(level=logging.INFO)
  click.echo(f"Debug mode is {'on' if debug else 'off'}")
  if debug:
    logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.pass_context
def train(ctx):
  click.echo('Training')

@cli.command()
@click.pass_context
def train_vae(ctx):
  pass

@cli.command()
@click.pass_context
def classify(ctx):
  pass

@cli.command()
@click.pass_context
def extract_features(ctx):

  logging.debug(f"Dataset: {dataset}")
  logging.debug(f"Debug: {debug}")
  logging.debug(f"Epochs: {epochs}")
  logging.debug(f"Batch Size: {batch_size}")
  logging.debug(f"Maximum batches per epoch: {max_batches}")
  logging.debug(f"Test-train split: {split*100}%")
  logging.debug(f"Base features: {n_base_features}")
  logging.debug(f"Latent features: {n_latent_features}")
  logging.debug(f"VAE Layers: {n_layers}")

  # TODO: define experiment name, make exp dir under predictions dir

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  data = HCSData.from_csv(dataset)  # Load dataset

  test_loader = torch.utils.data.DataLoader(  # Generate a testing loader
      data, batch_size=batch_size, shuffle=False)

  net = VAE_fm(lf=n_latent_features, base=n_base_features) # TODO: load from .pt
  logging.debug(net)

  if torch.cuda.device_count() > 1:  # If multiple gpu's
    net = torch.nn.DataParallel(net)  # Parallelize
  net.to(device)  # Move model to devic

  try:
    for epoch in range(epochs): # Iterate through epochs
      with torch.no_grad():
        for bn, (X, _) in tqdm(enumerate(test_loader), total=max_batches):

          x = X.to(device)
          o, u, logvar = net(x)

          X = x.cpu().detach().numpy()
          O = o.cpu().detach.numpy()
          err = mse(X, O)
          tqdm.write(err) # TODO: Format nicely

          # TODO: save feature maps (u, o) and predictions to exp dir
  except (KeyboardInterrupt, SystemExit):
    print("Session interrupted.")


if __name__ == '__main__':
  cli(obj={})
