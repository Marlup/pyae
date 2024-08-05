import os
from IPython.display import clear_output

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from matplotlib import pyplot as plt

from .constant import (
    RANDOM_STATE
)

from .evaluation import plot_reconstruction
from pyae.evaluation import plot_reconstruction, compute_losses_from_dataset
from pyae.utils import get_timestamp

################## 
#### Training ####
##################

def results_training_epoch(func):
    """
    Wrapper function to print the loss and learning rate during a training epoch.
    
    Args:
        func (function): The function being wrapped.

    Returns:
        function: The wrapper function.
    """
    def wrapper(*args, **kwargs):
        #optimizer = args[2]  # Assuming optimizer is always the third argument
        self = args[0]
        results = func(*args, **kwargs)
        
        # Print Loss and learning rate
        print(f"\tLearning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"\tTraining loss: {round(results, 6)}")
        return results
    
    return wrapper

def results_evaluation_epoch(func):
    """
    Wrapper function to print the loss during an evaluation epoch.
    
    Args:
        func (function): The function being wrapped.

    Returns:
        function: The wrapper function.
    """
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        
        # Print Loss
        print(f"\tEval loss: {round(results, 6)}")
        print(40 * "-")
        return results
    
    return wrapper

class TrainingManager:
    def __init__(
        self, 
        model, 
        train_loader,
        eval_loader=None,
        group_test_loaders=None,
        optimizer=None,
        criterion=None, 
        metrics=None, 
        lr_scheduler=None, 
        epochs=10, 
        tol=4e-5,
        max_no_improvements=5,
        checkpoint_frequency=5,
        model_directory="",
        T=10, 
        mode="standard",
        n_clusters=0, 
        postrain_config={},
        n_display_reset=6,
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.group_test_loaders = group_test_loaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.tol = tol
        self.max_no_improvements = max_no_improvements
        self.checkpoint_frequency = checkpoint_frequency
        if model_directory:
            self.model_directory = "../data/models/checkpoints"
        else:
            self.model_directory = model_directory
        self.T = T
        self.mode = mode
        self.n_clusters = n_clusters
        self.n_display_reset = n_display_reset

        # Initialize dictionary to save default training parameters
        if mode == "dcec":
            self._set_default_postrain_config(postrain_config)
        
        self.train_losses = []
        self.eval_losses = []
        self.p_target = None
        self.prev_loss = float("inf")
        self.no_improvement_count = 0

    def early_stopper(func):
        def wrapper(*args, **kwargs):
            #optimizer = args[2]  # Assuming optimizer is always the third argument
            self = args[0]
            results = func(*args, **kwargs)
            
            # Print Loss and learning rate
            print(f"\tLearning Rate: {self.optimizer.param_groups[0]['lr']}")
            print(f"\tTraining loss: {round(results, 6)}")
            return results
        
        return wrapper
    
    def train_model(self):
        # Run pretraining
        if self.mode == "dcec":
            self.pretrain_model()

        # Run main training
        self._train_model()

        # Run reset
        #self.reset_manager(True)
        
        # Run last training
        if self.mode == "dcec":
            self.postrain_model()

    def reset_manager(self, full_reset=True):
        self.p_target = 0
        self.prev_loss = float("inf")
        self.no_improvement_count = 0

        # Reset the entire model
        if full_reset:
            # Unfreeze autoencoder parameters
            self.model.unfreeze_ae_parameters()

            # Freeze clustering parameters
            self.model.freeze_clustering_parameters()
        
        # Reset optimizer
        from torch.optim import Adam
        
        optimizer_default_params = self.optimizer.defaults
        
        base_lr = self.postrain_config["base_lr"]
        weight_decay = self.postrain_config["weight_decay"]
        
        optimizer = Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        self.optimizer = optimizer
        
        # Reset learning rate scheduler
        if self.lr_scheduler is None:
            return None
        
        last_lr = self.lr_scheduler.get_last_lr()
        
        if base_lr > last_lr:
            from torch.optim.lr_scheduler import StepLR

            step_size,  = self.postrain_config["lr_schedulder_step_size"], 
            gamma = self.postrain_config["lr_scheduler_gamma"]
            
            self.lr_scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def _set_default_postrain_config(self, config):
        #if not isinstance(postrain_config, dict) or not postrain_config:

        # Setup default postrain configuration
        self.postrain_config = {}
        
        self.postrain_config["epochs"] = self.epochs
        
        optimizer_defaults = self.optimizer.defaults
        self.postrain_config["weight_decay"] = optimizer_defaults.get("weight_decay", 0.0)
        self.postrain_config["base_lr"] = optimizer_defaults.get("lr", 0.01)
        
        if self.lr_scheduler is not None:
            self.postrain_config["lr_scheduler_step_size"] = self.lr_scheduler.step_size
            self.postrain_config["lr_scheduler_gamma"] = self.lr_scheduler.gamma

        if self.mode == "dcec":
            self.postrain_config["gamma_clustering"] = self.criterion.gamma
            self.postrain_config["T"] = self.T
        self.postrain_config["early_stopping_tol"] = self.tol
        self.postrain_config["max_no_improvements"] = self.max_no_improvements
        
        # Update configuration dict with custom parameters
        if config:
            self.postrain_config.update(config)
    
    def kmeans_pretrain(self):
        from sklearn.cluster import KMeans
        n_clusters = self.model.clustering.n_clusters
        initializer = KMeans(n_clusters, n_init=20, random_state=RANDOM_STATE)

        # Retrieves unbatched latent data
        x = self._unbatch_outputs(index_select=1)
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
            
        initializer.fit(x)
        self.model.clustering.initialize_weights(torch.tensor(initializer.cluster_centers_))
        
    def pretrain_model(self):
        if self._should_update_p_target():
            self._update_p_target()

        self.model.freeze_clustering_parameters()

        # Deactivate clustering loss on criterion
        self.criterion.gamma = 0.0

    def postrain_model(self):
        # Freeze autoencoder parameters
        self.model.freeze_ae_parameters()
        
        # Unfreeze clustering parameters
        self.model.unfreeze_clustering_parameters()

        # Pretrain/Initialize clustering weights
        self.kmeans_pretrain()
        
        # Activate clustering loss on criterion
        self.criterion.gamma = self.postrain_config["gamma_clustering"]
        
        self._train_model()
    
    def _train_model(self):
        on_early_stopping = False
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(40 * "-")
            
            # Train one epoch
            epoch_loss = self._train_epoch()
            self.train_losses.append(epoch_loss)

            # Save model at checkpoint
            if epochs % self.checkpoint_frequency == 0:
                self._save_state(epoch, loss)
            
            # Learning rate scheduler step
            if self.lr_scheduler:
                self.lr_scheduler.step()
                
            # Evaluate one epoch
            # Run one step of early_stopping
            if self.eval_loader is not None:
                eval_loss = self._eval_epoch()
                self.eval_losses.append(eval_loss)
                
                on_early_stopping = self._early_stopping(eval_loss)
            
            # Check for early stopping
            if on_early_stopping:
                break
            
            self.prev_loss = epoch_loss
            
            # Clear output if required
            if epoch > 0 and epoch % self.n_display_reset == 0:
                clear_output()
        
        self.model.eval()
    
    @results_training_epoch
    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        
        # Check if we need to update p_target with all available examples
        #if self.p_target is None and self._should_update_p_target():
        if self._should_update_p_target():
            self._update_p_target()
        
        for i, batch in enumerate(self.train_loader):
            # Reset gradients for a new batch
            self.optimizer.zero_grad()
            
            # Compute forward step and loss
            loss = self._compute_batch_loss(batch)
            
            # Compute backpropagation
            self._compute_graph_gradients(loss)
            
            # Update weights and parameters
            self._update_parameters()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader.dataset)
    
    @results_evaluation_epoch
    def _eval_epoch(self):
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                loss = self._compute_batch_loss(batch)
                epoch_loss += loss.item()
        
        return epoch_loss / len(self.eval_loader.dataset)

    def _compute_batch_loss(self, batch, **kwargs):
        x, target = batch["x"], batch["y"]
        
        if self.mode == "standard":
            x_categories = batch["x_categories"]
        else:
            x_categories = None
        
        if self.mode == "standard":
            return self._compute_batch_loss_standard(x, x_categories, target, **kwargs)
        elif self.mode == "stack":
            return self._compute_batch_loss_stack(x, x_categories, target, **kwargs)
        elif self.mode == "vae":
            return self._compute_batch_loss_vae(x, x_categories, target, **kwargs)
        elif self.mode == "dcec":
            return self._compute_batch_loss_dcec(x, x_categories, target, **kwargs)
        elif self.mode == "classification":
            return self._compute_batch_classification(x, target, **kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def _compute_batch_loss_standard(self, x, x_categories, target, return_outputs=False):
        outputs = self.model(x, x_categories)
        loss = self.criterion(outputs, target) 
        
        if return_outputs:
            return loss.cpu(), outputs.cpu()
        return loss
        
    def _compute_batch_loss_stack(self, x, x_categories, target, return_outputs=False):
        outputs, target = self.model(x, x_categories)
        loss = self.criterion(outputs, target) 
        
        if return_outputs:
            return loss.cpu(), outputs.cpu()
        return loss
        
    def _compute_batch_loss_vae(self, x, x_categories, target, return_outputs=False):
        outputs, mean, log_var = self.model(x, x_categories)
        loss = self.criterion(outputs, target, mean, log_var)
        
        if return_outputs:
            return loss.cpu(), outputs.cpu()
        return loss
    
    def _compute_batch_loss_dcec(self, x, x_categories, target, return_outputs=False):
        outputs, z, q_dist = self.model(x, x_categories)
        #on_update_p_target = self._should_update_p_target()
        #if on_update_p_target and \
        #  (self.p_target is None or (i % self.T == 0) or q_dist.shape != self.p_target.shape):
        #    self._update_p_target()
        loss = self.criterion(outputs, target, q_dist, self.p_target)
        
        if return_outputs:
            return loss.cpu(), outputs.cpu()
        return loss
    
    def _compute_batch_classification(self, x, target, return_outputs=False):
        outputs = self.model(x)
        
        loss = self.criterion(outputs, target)
        
        if return_outputs:
            return loss.cpu(), outputs.cpu()
        return loss
    
    def _compute_graph_gradients(self, loss):
        loss.backward()
        
    def _update_parameters(self):
        self.optimizer.step()
    
    def _should_update_p_target(self):
        return self.T > 0 and len(self.train_loader.dataset) % self.T == 0
    
    def _update_p_target(self):
        print("Updating target distribution...")
        all_q_dists = self._unbatch_outputs(index_select=-1)
        all_q_dists = torch.cat(all_q_dists, dim=0)
        self.p_target = self.model.get_target_distribution(all_q_dists)
        self.model.train()

    def _unbatch_outputs(self, dataloader=None, dim=0, index_select=None):
        unbatched_outputs = []
        self.model.eval()

        if dataloader is None:
            dataloader = self.train_loader
        
        with torch.no_grad():
            for batch in dataloader:
                x, x_categories = batch["x"], batch["x_categories"]
                
                outputs = self.model(x, x_categories)
                
                if index_select is None:
                    unbatched_outputs.append(outputs)
                else:
                    # Select a specific element of the iterable output
                    unbatched_outputs.append(outputs[index_select])
        
        # Concatenate all unbatched_outputs to form a complete set
        return torch.cat(unbatched_outputs, dim=dim)

    def _early_stopping(self, current_loss):
        """
        Returns True if there weren't sufficient improvements,
        otherwise False.
        """
        
        loss_change = abs(self.prev_loss - current_loss)
        
        if loss_change < self.tol:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        
        if self.no_improvement_count >= self.max_no_improvements:
            self.no_improvement_count = 0
            print(f"Early stopping after {self.max_no_improvements} epochs: Loss improvement threshold reached.")
            return True
        return False

    def _save_state(self, epoch, loss):
        model_name = f"model_checkpoint_epoch_{ts}_at_{get_timestamp()}.pt"
        model_path = os.path.join(self.model_directory, model_name)
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, 
            model_path
        )

    def _load_state(self, model_name, on_eval=True):
        model_path = os.path.join(self.model_directory, model_name)
        checkpoint = torch.load(model_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if on_eval:
            self.model.eval()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get("epoch", None), checkpoint.get("loss", None)
        
    def evaluate_model(self):
        self.model.eval()
        eval_loss = 0.0
        eval_losses = []
        
        with torch.no_grad():
            for batch in self.eval_loader:
                loss = self._eval_epoch()
                eval_loss += loss
                eval_losses.append(loss)
        
        avg_eval_loss = eval_loss / len(self.eval_loader.dataset)
        return avg_eval_loss

    def plot_losses(self):
        if self.mode == "dcec":
            epochs = self.postrain_config["epochs"]
            learning_rate = self.postrain_config["base_lr"]
            gamma = self.postrain_config["lr_scheduler_gamma"]
            step_size = self.postrain_config["lr_scheduler_step_size"]
        else:
            epochs = self.epochs
            learning_rate = self.base_lr
            gamma = self.gamma
            step_size = self.step_size
            
        plot_losses(
            self.train_losses, 
            self.eval_losses, 
            epochs=epochs,
            learning_rate=learning_rate
            gamma=gamma
            step_size=step_size
            )

    def report_model_evaluation(self, kind_plot="bar", figsize=(12, 8), on_return_dataframe=True):
        metrics_losses = []
        dataloaders_label = []

        # Compute the metric scores/losses for each dataframe provided
        for dataloader_label, test_loader in self.group_test_loaders.item():
            metric_losses = compute_losses_from_dataset(self.training_manager, test_loader, self.metrics)
            
            metrics_losses.append(metric_losses)
            dataloaders_label.append(dataloader_label)

        # Build a dataframe for the metrics
        metrics_by_dataloader = pd.DataFrame(metrics_losses,
                                             index=dataloaders_label) \
                                  .T
        # Plot the metrics
        metrics_by_dataloader.plot(kind=kind_plot, rot=30, figsize=figsize)
        
        if on_return_dataframe:
            return metrics_by_dataloader

def plot_losses(train_losses, eval_losses=None, **kwargs):
    fig = plt.figure()
    y_lim = kwargs.get("y_lim", None)
    epochs = kwargs.get("epochs", None)
    learning_rate = kwargs.get("learning_rate", None)
    gamma = kwargs.get("gamma", None)
    step_size = kwargs.get("step_size", None)
    color = kwargs.get("color", "red")
    
    plt.plot(train_losses, label='Training Loss')
    if eval_losses:
        plt.plot(eval_losses, label='Validation Loss')
    
    if learning_rate and gamma and step_size and epochs:
        for position, _ in enumerate(get_exp_adaptive_learning(epochs, learning_rate, gamma, step_size)):
            x = step_size * (position + 1)
            plt.axvline(x, 0.0, 1.0, color=color, alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.ylim(y_lim)
    plt.legend()
    plt.show()
    return fig

def run_experiment(
    path_experiment_specs, 
    model_name,
    autoencoder_class,
    input_length, 
    train_loader, 
    test_loader_train=None, 
    test_loader_moderate=None, 
    test_loader_hard=None, 
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
    """
    Run an experiment defined by the experiment specifications.

    Args:
        path_experiment_specs (str): Path to the experiment specifications file.
        model_name (str): Name of the model.
        autoencoder_class (class): Autoencoder class.
        input_length (int): Length of the input data.
        train_loader (DataLoader): DataLoader for training data.
        test_loader_train (DataLoader, optional): DataLoader for training data for testing. Defaults to None.
        test_loader_moderate (DataLoader, optional): DataLoader for moderate load testing data. Defaults to None.
        test_loader_hard (DataLoader, optional): DataLoader for hard load testing data. Defaults to None.
        input_channel (int, optional): Number of input channels. Defaults to 1.
        specs_additional (dict, optional): Additional experiment specifications. Defaults to None.
        n_categories (int, optional): Number of categories. Defaults to 0.
        epochs (int, optional): Number of epochs for training. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 0.1.
        l2_amount (float, optional): L2 regularization amount. Defaults to 0.0.
        step_size (int, optional): Step size for learning rate scheduler. Defaults to 10.
        gamma (float, optional): Gamma value for learning rate scheduler. Defaults to 0.5.
        criterion (str, optional): Criterion for loss calculation. Defaults to "mse".
        n_reconstructions (int, optional): Number of reconstructions for testing data. Defaults to 20.
        on_return_losses (bool, optional): Flag to return losses. Defaults to False.
        tol (float, optional): Tolerance for early stopping. Defaults to 1e-4.
        no_improvement_count (int, optional): Number of improvements before early stopping the optimitization algo.
        T (int, optional): Number of iterations to update the target distribution. Default: 10.
        mode (str, optional): Training mode. Default: "regular".
        n_clusters (int, optional): Number of clusters for 'dcec' mode..

    Returns:
        tuple: If on_return_losses is True, returns a tuple containing losses_experiment (dict) and models (list).
    """
    
    # Define loss class
    if criterion == "mse":
        criterion = nn.MSELoss()
    elif criterion == "maPe":
        from torchmetrics import MAPE
        criterion = nn.MAPE()
    elif criterion == "mae":
        from torchmetrics import MAE
        criterion = nn.MAE()
    
    # Load experiment specifications
    with open(path_experiment_specs, "r") as file:
        experiment_specs = load(file)
    
    losses_experiment = {}
    models = []
    
    for specs_desc, specs in experiment_specs.items():
        path_autoencoder = os.path.join("models", model_name + "_" + specs_desc + ".pt")
        
        # Skip training if the model already exists
        if os.path.exists(path_autoencoder):
            print(f"Model {path_autoencoder} exists. Skip to the next iteration")
            continue
        
        if specs_additional:
            specs.update(specs_additional)
        
        # Define autoencoder
        autoencoder_inputs = (
            input_length,
            input_channel,
            specs.get("encoder_specs", None),
            specs.get("latent_specs", None),
            specs.get("decoder_specs", None),
        )
        autoencoder_model = autoencoder_class(*autoencoder_inputs, n_categories=n_categories)
        print(autoencoder_model.summarize_model((input_channel, input_length)))
        
        # Define optimizer
        optimizer = Adam(autoencoder_model.parameters(), lr=lr, weight_decay=l2_amount)
        
        # Define scheduler
        lr_scheduler = StepLR(optimizer, step_size, gamma=gamma)

        # Initialize the TrainingManager
        training_manager = TrainingManager(
            model=autoencoder_model, 
            train_loader=train_loader, 
            eval_loader=None,  # Assuming no eval_loader is provided
            optimizer=optimizer, 
            criterion=criterion, 
            lr_scheduler=lr_scheduler, 
            epochs=epochs, 
            tol=tol, 
            no_improvement_count=10,
            T=T,
            mode=mode,
            n_clusters=n_clusters,
        )
        
        # Train model
        train_losses, eval_losses = training_manager.train_model()
        
        # Plot loss evolution
        plot_losses(train_losses, eval_losses, lr=lr, gamma=gamma, step_size=step_size)
        image_path = os.path.join("images", f"training_loss_evolution_{specs_desc}.png")
        plt.savefig(image_path)

        # Plot reconstructions for different test loaders
        if test_loader_train is not None:
            plot_reconstruction(test_loader_train, autoencoder_model)
            image_path = os.path.join("images", f"testing_moderate_load_reconstructions_{n_reconstructions}_{specs_desc}.png")
            plt.savefig(image_path)
        
        if test_loader_moderate is not None:
            plot_reconstruction(test_loader_moderate, autoencoder_model)
            image_path = os.path.join("images", f"testing_moderate_load_reconstructions_{n_reconstructions}_{specs_desc}.png")
            plt.savefig(image_path)
        
        if test_loader_hard is not None:
            plot_reconstruction(test_loader_hard, autoencoder_model)
            image_path = os.path.join("images", f"testing_high_load_reconstructions_{n_reconstructions}_{specs_desc}.png")
            plt.savefig(image_path)
        
        if on_return_losses:
            losses_experiment[specs_desc] = (train_losses, eval_losses)
        
        # Save the model
        torch.save(autoencoder_model, path_autoencoder)
        models.append(autoencoder_model)
    
    if on_return_losses:
        return losses_experiment, models

#########
# UTILS #
#########

def get_exp_adaptive_learning(epochs, lr=0.1, gamma=0.5, step_size=10):
    if epochs >= step_size:
        n_steps = epochs // step_size
        #print([(f"Reduction {i}", lr * gamma**i) for i in range(n_reductions)])
        return [lr * gamma**i for i in range(n_steps)]
    return []