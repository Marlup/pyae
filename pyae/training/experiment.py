# runner.py

import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MAE, MAPE
import matplotlib.pyplot as plt
from yaml import safe_load as load

from evaluation import plot_reconstruction
from utils.plotting import plot_losses
from training.training import TrainingManager


def run_experiment(
    path_experiment_specs, 
    model_name,
    autoencoder_class,
    input_length, 
    train_loader, 
    test_loaders=None,
    input_channel=1,
    specs_additional=None,
    n_categories=0,
    epochs=10, 
    lr=0.1, 
    l2_amount=0.0, 
    step_size=10, 
    gamma=0.5, 
    criterion="mse",
    n_reconstructions=20,
    on_return_losses=False,
    tol=1e-4,
    max_no_improvements=5,
    T=100,
    mode="standard",
    n_clusters=0,
    postrain_config={},
):
    # Define loss
    if criterion == "mse":
        criterion = nn.MSELoss()
    elif criterion == "mae":
        criterion = MAE()
    elif criterion == "mape":
        criterion = MAPE()

    # Load specs
    with open(path_experiment_specs, "r") as file:
        experiment_specs = load(file)

    losses_experiment = {}
    models = []

    for specs_desc, specs in experiment_specs.items():
        if specs_additional:
            specs.update(specs_additional)

        model_path = os.path.join("models", f"{model_name}_{specs_desc}.pt")
        if os.path.exists(model_path):
            print(f"Model {model_path} already exists. Skipping...")
            continue

        ae_model = autoencoder_class(
            input_length,
            input_channel,
            specs.get("encoder_specs"),
            specs.get("latent_specs"),
            specs.get("decoder_specs"),
            n_categories=n_categories
        )
        print(ae_model.summarize_model((input_channel, input_length)))

        optimizer = Adam(ae_model.parameters(), lr=lr, weight_decay=l2_amount)
        scheduler = StepLR(optimizer, step_size, gamma)

        
        manager = TrainingManager(
            model=ae_model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=scheduler,
            epochs=epochs,
            tol=tol,
            max_no_improvements=max_no_improvements,
            T=T,
            mode=mode,
            n_clusters=n_clusters,
            postrain_config=postrain_config,
        )

        manager.train_model()

        # Plot losses
        plot_losses(manager.train_losses, manager.eval_losses, lr, gamma, step_size)
        fig_path = os.path.join("images", f"losses_{specs_desc}.png")
        plt.savefig(fig_path)
        plt.close()

        # Plot reconstructions
        if test_loaders:
            for label, test_loader in test_loaders.items():
                plot_reconstruction(test_loader, ae_model, n_reconstructions=n_reconstructions)
                recon_path = os.path.join("images", f"recons_{label}_{specs_desc}.png")
                plt.savefig(recon_path)
                plt.close()

        if on_return_losses:
            losses_experiment[specs_desc] = (manager.train_losses, manager.eval_losses)

        torch.save(ae_model, model_path)
        models.append(ae_model)

    return (losses_experiment, models) if on_return_losses else models
