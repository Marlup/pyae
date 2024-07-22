import torch
from matplotlib import pyplot as plt

def plot_reconstruction(
    dataloader, 
    training_manager, 
    n_reconstructions=20, 
    n_columns=4, 
    width=12, 
    height_per_row=3, 
    digits=4, 
    suptitle="Reconstruction of signals"
):
    """
    Plot original and reconstructed signals.

    Args:
        dataloader (DataLoader): DataLoader containing the signals.
        training_manager (nn.Module): TrainingManager which contains the model and training functionalities.
        n_reconstructions (int, optional): Number of signals to reconstruct. Default is 20.
        n_columns (int, optional): Number of columns in the plot grid. Default is 4.
        width (int, optional): Width of the figure. Default is 12.
        height_per_row (int, optional): Height per row in the figure. Default is 3.
        digits (int, optional): Number of digits to round the loss. Default is 4.
        suptitle (str, optional): Title of the figure. Default is "Reconstruction of signals".
    """
    if dataloader.batch_size > 1:
        raise Exception(f"Argument error. 'dataloader' batch_size should be 1. \
Current batch_size is {dataloader.batch_size}")

    # Set model to evaluation mode
    training_manager.model.eval()
    
    # Make subplots
    n_rows = n_reconstructions // n_columns
    figsize = (width, height_per_row * n_rows)
    fig, axes = plt.subplots(n_reconstructions // n_columns, 
                             n_columns, 
                             figsize=figsize)
    # Make axes a 1d-array for ease of use
    axes = axes.ravel()
    
    for i, batch in enumerate(dataloader):
        if i >= n_reconstructions:
            break

        ## Setup data
        x, y = batch["x"], batch["y"]
        x_categories = batch.get("x_categories", None)
        ids = batch["ids"]
        
        # Squeeze target and reconstruction (prediction) into 1d-array
        deviation, reconstruction = training_manager._compute_batch_loss(x, x_categories, y, return_outputs=True)
        
        # Plot original and reconstructed signals
        axes[i].plot(y.cpu().detach().numpy().squeeze(), label=f"Original", color="blue")
        axes[i].plot(reconstruction.cpu().detach().numpy().squeeze(), label="Predicted", color="green")
        
        # Setup labels, title and embellishments
        title = f"Loss: {round(deviation.cpu().item(), digits)}"
        
        if ids:
            ids_format = "load {}, sweep {}, sensor {}".format(*ids)
            title += f";\nSignal id: {ids_format}"
        
        axes[i].set_title(title)
        axes[i].set_xlabel("Frequency")
        axes[i].set_ylabel("Impedance")
        axes[i].legend(fontsize=8)
    
    plt.suptitle(suptitle)
    plt.tight_layout()

def plot_ci_reconstruction(
    dataloader, 
    training_manager,
    models_ci,
    n_reconstructions=20, 
    n_columns=4, 
    width=12, 
    height_per_row=3, 
    digits=4, 
    suptitle="Reconstruction of signals"
):
    """
    Plot original and reconstructed signals with confidence intervals.

    Args:
        dataloader (DataLoader): DataLoader containing the signals.
        training_manager_reconstruction (nn.Module): TrainingManager which contains the model and training functionalities.
        models_ci (tuple of nn.Modules): Models used for computing lower and upper confidence intervals.
        n_reconstructions (int, optional): Number of signals to reconstruct. Default is 20.
        n_columns (int, optional): Number of columns in the plot grid. Default is 4.
        width (int, optional): Width of the figure. Default is 12.
        height_per_row (int, optional): Height per row in the figure. Default is 3.
        digits (int, optional): Number of digits to round the loss. Default is 4.
        suptitle (str, optional): Title of the figure. Default is "Reconstruction of signals".
    """
    if dataloader.batch_size > 1:
        raise Exception(f"Argument error. 'dataloader' batch_size should be 1. \
Current batch_size is {dataloader.batch_size}")
    
    # Set model to evaluation mode
    model_reconstruction.eval()
    
    # Make subplots
    n_rows = n_reconstructions // n_columns
    figsize = (width, height_per_row * n_rows)
    fig, axes = plt.subplots(n_reconstructions // n_columns, 
                             n_columns, 
                             figsize=figsize)
    axes = axes.ravel()
    
    for i, batch in enumerate(dataloader):
        if i >= n_reconstructions:
            break
        
        # Setup data
        x, y = batch["x"], batch["y"]
        x_categories = batch.get("x_categories", None)
        
        # Squeeze target and reconstruction (prediction) into 1d-array; ## Compute losses
        deviation, reconstruction = training_manager._compute_batch_loss(x, x_categories, y, return_outputs=True)
        deviation = deviation.item()
        reconstruction = reconstruction.cpu().detach().squeeze()
        
        # Lower confidence interval band
        lower_recon = models_ci[0](x, *x_categories).detach()
        lower_recon = lower_recon.cpu().squeeze()
        # Upper confidence interval band
        upper_recon = models_ci[1](x, *x_categories).detach()
        upper_recon = upper_recon.cpu().squeeze()
        
        # Plot original and reconstruction signals with confidence intervals
        y = y.squeeze()
        axes[i].plot(y, label=f"Original", color="blue")
        axes[i].plot(reconstruction, label="Predicted", color="green")
        axes[i].plot(lower_recon, label="Lower band", color="cyan")
        axes[i].plot(upper_recon, label="Upper interval", color="red")
        
        # Setup labels, title, and embellishments
        title = f"Loss: {round(deviation, digits)}"
        axes[i].set_title(title)
        axes[i].set_xlabel("Frequency")
        axes[i].set_ylabel("Impedance")
        axes[i].legend(fontsize=8)
    
    plt.suptitle(suptitle)
    plt.tight_layout()

def plot_model_reconstruction_deviation(
    df_loss,
    signals_xarray,
    default_sweep, 
    subplot_column, 
    loss_column, 
    x_column, 
    sweep_column,
    loss_as_percentage=False,
    width=10,
    height=25,
    legend_fontsize=5,
    fig_facecolor="grey"
):
    n_cols = 2
    boxplot_y_label = loss_column.upper()
    sensors = df_loss[subplot_column].unique()
    n_sensors = df_loss[subplot_column].nunique()
    
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(f"{boxplot_y_label} deviation between incoming signal and model reconstruction\n (The signals are scaled at [0, 1])", y=1.0)
    fig.set_facecolor(fig_facecolor)
    
    for idx_sensor, sensor in enumerate(sensors):
        # Indices
        subplot_idx = n_cols * idx_sensor + 1
        
        df_loss_by_subplot_col = df_loss[df_loss[subplot_column] == sensor]
        
        ys = []
        for group, data_group in df_loss_by_subplot_col.groupby(x_column):
            # Prepare the data
            try:
                data_to_plot = signals_xarray.sel({subplot_column: sensor, x_column: group, sweep_column: default_sweep})
            except:
                continue
            
            #Plot the signals
            plt.subplot(n_sensors, n_cols, subplot_idx)
            plt.plot(data_to_plot, label=f"{x_column} {group}")
            
            # x-y labels
            plt.xlabel("frequency")
            plt.ylabel("Real impedance")
            
            # Store data for upcoming boxplot representation
            loss_values = data_group[loss_column].values
            
            if loss_as_percentage:
                ys.append(100 * loss_values)
            else:
                ys.append(loss_values)
                
        plt.legend(fontsize=legend_fontsize)
        
        # Box plot of MSE deviations
        plt.subplot(n_sensors, n_cols, subplot_idx + 1)
        plt.boxplot(ys)
        
        # x-y labels
        plt.xlabel(x_column)
        plt.ylabel(boxplot_y_label + " (%)" if loss_as_percentage else "")
        
        # Title label
        plt.title(f"{subplot_column.capitalize()} {sensor}")
        
    plt.tight_layout()

def compute_losses_from_dataset(training_manager, dataloader, losses, on_vae=False):
    """
    Compute losses from a dataset using a model and specified loss functions.

    Args:
        training_manager (nn.Module): TrainingManager which contains the model and training functionalities.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        losses (dict): Dictionary with keys as the name of the loss and values as loss instances.
        on_vae (bool, optional): Whether the model is a Variational Autoencoder (VAE). Default is False.

    Returns:
        dict: Dictionary of loss statistics with keys as concatenated loss name and statistic (e.g., "loss_mean").
    """
    training_manager.model.eval()
    deviations = {}

    for name, loss in losses.items():
        deviation = get_loss_statistics(training_manager, dataloader, loss, on_vae=on_vae)
        deviations.update({f"{name}_{k}": v for k, v in deviation.items()})
    return deviations

def compute_loss(training_manager, dataloader, loss_func, on_vae):
    """
    Compute losses for a model and a dataset using a specific loss function.

    Args:
        training_manager (nn.Module): TrainingManager which contains the model and training functionalities.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        loss_func (torch.nn.Module): Loss function instance.
        on_vae (bool): Whether the model is a Variational Autoencoder (VAE).

    Returns:
        torch.Tensor: Tensor containing the losses.
    """
    losses = []
    for batch in dataloader:
        # Setup data
        x, y = batch["x"], batch["y"]
        x_categories = batch.get("x_categories", None)
        
        loss = training_manager._compute_batch_loss(x, x_categories, y)

        losses.append(loss)
    return torch.tensor(losses, dtype=torch.float32)

def get_loss_statistics(training_manager, dataloader, loss_func, on_vae=False):
    """
    Compute statistics (mean, standard deviation, median, and IQR) of losses.

    Args:
        training_manager (nn.Module): TrainingManager which contains the model and training functionalities.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        loss_func (torch.nn.Module): Loss function instance.
        on_vae (bool, optional): Whether the model is a Variational Autoencoder (VAE). Default is False.

    Returns:
        dict: Dictionary of loss statistics with keys 'mean', 'std', 'median', and 'iqr'.
    """
    losses = compute_loss(training_manager, dataloader, loss_func, on_vae=on_vae)
    return {
        "mean": losses.mean(),
        "std": losses.std(),
        "median": losses.median(),
        "iqr": losses.quantile(0.75) - losses.quantile(0.25)
    }

def get_confusion_matrix(X, y, model):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    y_true = y.cpu().detach().numpy()
    y_pred = model(X).cpu().detach().numpy().argmax(axis=1)
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return conf_matrix, ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

def get_top_k_accuracy(X, y, model, k=2):
    from sklearn.metrics import top_k_accuracy_score
    
    y_true = y.cpu().detach().numpy()
    y_pred = model(X).cpu().detach().numpy()

    score = top_k_accuracy_score(y_true, y_pred, k=k)
    return score

def get_classification_report(X, y, model):
    from sklearn.metrics import classification_report
    
    y_true = y.cpu().detach().numpy()
    y_pred = model(X).cpu().detach().numpy().argmax(-1)

    report = classification_report(y_true, y_pred)    
    return report

def get_top_k_categories(X, model, k=2):
    y_pred = model(X)
    top_k_values, top_k_indices = torch.topk(y_pred, k, dim=-1)
    
    return top_k_values.cpu(), top_k_indices.cpu()