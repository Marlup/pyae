import torch
from matplotlib import pyplot as plt
from pyae.architecture import AutoencoderLayerBuilder

class EvaluationManager(AutoencoderLayerBuilder):
    def __init__(
        self, 
        model, 
        train_loader,
        test_loader,
        metrics,
        mode="classification",
        cv=1,

    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.metrics = metrics
        self.mode = mode
        self.cv = cv
        
        # Activated cross validation
        if cv > 1:
            self.kfold_models = []
            self.kfold_train_losses = []
            self.kfold_test_losses = []


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
        raise Exception(f"Argument error. batch_size should be 1. \
Current batch_size is {dataloader.batch_size}")

    # Set model to evaluation mode
    training_manager.model.eval()

    # Make subplots
    n_rows = n_reconstructions // n_columns
    figsize = (width, height_per_row * n_rows)
    _, axes = plt.subplots(n_reconstructions // n_columns,
                             n_columns,
                             figsize=figsize)
    # Make axes a 1d-array for ease of use
    axes = axes.ravel()

    for i, batch in enumerate(dataloader):
        if i >= n_reconstructions:
            break

        # Squeeze target and reconstruction (prediction) into 1d-array
        deviation, reconstruction = training_manager._compute_batch_loss(batch, return_outputs=True)

        # Plot original and reconstructed signals
        axes[i].plot(batch["y"].cpu().detach().numpy().squeeze(), label="Original", color="blue")
        axes[i].plot(reconstruction.cpu().detach().numpy().squeeze(), label="Predicted", color="green")

        # Setup labels, title and embellishments
        title = f"Loss: {round(deviation.cpu().item(), digits)}"
        if batch["ids"]:
            ids_format = "load {}, sweep {}, sensor {}".format(*batch["ids"])
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
        raise Exception(f"Argument error. batch_size should be 1. \
Current batch_size is {dataloader.batch_size}")
    
    # Set model to evaluation mode
    training_manager.model.eval()
    
    # Make subplots
    n_rows = n_reconstructions // n_columns
    figsize = (width, height_per_row * n_rows)
    _, axes = plt.subplots(n_reconstructions // n_columns, 
                             n_columns, 
                             figsize=figsize)
    axes = axes.ravel()
    
    for i, batch in enumerate(dataloader):
        if i >= n_reconstructions:
            break
        
        # Squeeze target and reconstruction (prediction) into 1d-array; ## Compute losses
        deviation, reconstruction = training_manager._compute_batch_loss(batch, return_outputs=True)
        deviation = deviation.item()
        reconstruction = reconstruction.cpu().detach().squeeze()
        
        # Lower confidence interval band
        lower_recon = models_ci[0](batch["x"]).detach()
        lower_recon = lower_recon.cpu().detach().squeeze()
        # Upper confidence interval band
        upper_recon = models_ci[1](batch["x"]).detach()
        upper_recon = upper_recon.cpu().detach().squeeze()
        
        # Plot original and reconstruction signals with confidence intervals
        y = y.squeeze()
        axes[i].plot(batch["y"], label="Original", color="blue")
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

def compute_losses_from_dataloader(training_manager, dataloader, losses, on_vae=False):
    """
    Compute losses from a dataset using a model and specified loss functions.

    Args:
        training_manager (nn.Module): TrainingManager which contains the model and training functionalities.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        losses (tuple): Tuple with name of loss and loss in that specific order.
        on_vae (bool, optional): Whether the model is a Variational Autoencoder (VAE). Default is False.

    Returns:
        dict: Dictionary of loss statistics with keys as concatenated loss name and statistic (e.g., "loss_mean").
    """
    training_manager.model.eval()
    losses_statistics = {}

    for loss_name, loss_func in losses.items():
        loss_statistics = get_loss_statistics(training_manager, dataloader, loss_func, on_vae=on_vae)
        losses_statistics.update({f"{loss_name}_{stat_name}": stat_value.item() for stat_name, stat_value in loss_statistics.items()})
    
    return losses_statistics

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
        loss = training_manager._compute_batch_loss(batch)

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

def get_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    return confusion_matrix(y_true, y_pred), ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

def get_top_k_accuracy(y_true, y_pred, k=2):
    from sklearn.metrics import top_k_accuracy_score
    return top_k_accuracy_score(y_true, y_pred, k=k)

def get_classification_report(y_true, y_pred=None):
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred)

def get_top_k_categories(y_pred, k=2, dim=-1):
    top_k_values, top_k_indices = torch.topk(y_pred, k, dim=dim)
    return top_k_values, top_k_indices

def transform_prob_prediction(y_pred_prob, k, on_hard_prediction=True):
    from numpy import ndarray, array
    if not isinstance(y_pred_prob, ndarray):
        y_pred_prob = array(y_pred_prob)
    
    if on_hard_prediction:
        return y_pred_prob.argmax(axis=1)
    else:
        return y_pred_prob.argsort()[:, :-k-1:-1]

def k_neighbors_confusion_matrix(y_test, y_pred):
    from numpy import zeros_like, diag, eye
    # Get standard confusion matrix
    conf_matrix, _ = get_confusion_matrix(y_test, y_pred)
    
    # Extract the main diagonal from confusion matrix
    main_diag = diag(conf_matrix)

    # Extract lower triangle matrix
    adjust_first_low_diag = zeros_like(main_diag)
    adjust_first_low_diag[1:] = diag(conf_matrix, -1)

    # Extract upper triangle matrix
    adjust_first_up_diag = zeros_like(main_diag)
    adjust_first_up_diag[:-1] = diag(conf_matrix, 1)

    n = len(conf_matrix)
    indices_first_up = eye(n, k=1, dtype=bool)
    indices_first_low = eye(n, k=-1, dtype=bool)
    indices_diag = eye(n, k=0, dtype=bool)

    # Adjust the k + 1 diagonals of the confusion matrix
    adjusted_conf_matrix = conf_matrix.copy()
    adjusted_conf_matrix[indices_first_up] = 0
    adjusted_conf_matrix[indices_first_low] = 0
    adjusted_conf_matrix[indices_diag] += adjust_first_low_diag + adjust_first_up_diag
    
    return adjusted_conf_matrix

def k_neighbors_classification_report_from_confusion_matrix(cf, opt_xlabel="load"):
    from numpy import diag, arange

    n = len(cf)

    true_pos = diag(cf, 0)
    false_neg = cf.sum(axis=1) - true_pos
    false_pos = cf.sum(axis=0) - true_pos

    print(f"True positives {true_pos}\nFalse negatives: {false_neg}\nFalse positives: {false_pos}")

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1_score = 2 * precision * recall /  (precision + recall)
    x = 1.5 * arange(n)

    plt.figure(figsize=(9, 6))
    plt.bar(x, 100 * precision, align="center", width=0.25, label="precision (%)")
    plt.bar(x + 0.25, 100 * recall, align="center", width=0.25, label="recall (%)")
    plt.bar(x + 2 * 0.25, 100 * f1_score, align="center", width=0.25, label="f1 (%)")
    plt.axhline(50, color="k", alpha=0.2, linestyle="--")
    plt.axhline(70, color="k", alpha=0.6, linestyle="--")
    plt.axhline(90, color="k", alpha=1.0, linestyle="--")

    plt.xticks(x, arange(n))
    plt.title("Evaluation on first-neighbor predictions")
    plt.xlabel(f"class / {opt_xlabel}")
    plt.ylabel("score")
    plt.legend(fontsize="x-small")

    return precision, recall, f1_score

def ranker_loss(y_pred, y_test):
    from torchmetrics import(
        MeanAbsoluteError,
        MeanSquaredError,
        MeanAbsolutePercentageError,
        ExplainedVariance,
        R2Score
    )
    from numpy import mean, std, linspace

    mae_metric = MeanAbsoluteError()
    mrse_metric = MeanSquaredError(squared=False)
    mape_metric = MeanAbsolutePercentageError()
    ev_metric = ExplainedVariance(multioutput='raw_values')
    r2_metric = R2Score()

    mae_scores = [mae_metric(yp, yt).item() for yp, yt in zip(y_pred, y_test)]
    mae_mean_scores = mean(mae_scores)
    mae_std_scores = std(mae_scores)

    mrse_scores = [mrse_metric(yp, yt).item() for yp, yt in zip(y_pred, y_test)]
    mrse_mean_scores = mean(mrse_scores)
    mrse_std_scores = std(mrse_scores)

    mape_scores = [mape_metric(yp, yt).item() for yp, yt in zip(y_pred, y_test)]
    mape_mean_scores = mean(mape_scores)
    mape_std_scores = std(mape_scores)

    print(f"Mean and std mae: {mae_mean_scores, mae_std_scores}")
    print(f"Mean and std mrse: {mrse_mean_scores, mrse_std_scores}")
    print(f"Mean and std mape: {mape_mean_scores, mape_std_scores}")
    print(f"ExplainedVariance and std mape: {100 * ev_metric(y_pred, y_test)}")
    print(f"R2Score: {100 * r2_metric(y_pred, y_test)}")

    plt.figure(figsize=(8, 8))

    # First row
    plt.subplot(3, 3, 1)
    _ = plt.boxplot([mae_scores], bootstrap=10000, meanline=True)
    plt.title("MAE")

    plt.subplot(3, 3, 2)
    _ = plt.boxplot([mape_scores])
    plt.title("MAPE")

    plt.subplot(3, 3, 3)
    _ = plt.boxplot([mrse_scores])
    plt.title("MRSE")

    # Second row
    plt.subplot(3, 1, 2)
    residuals = y_pred - y_test
    _ = plt.plot(residuals, marker="o", linestyle="", markersize=1.5)
    plt.axhline(0, color="k", linestyle="--", alpha=0.6)

    plt.title("Scatter of residuals")
    ticks = linspace(0, len(mape_scores), 10, dtype=int)
    plt.xticks(ticks=ticks, labels=range(10))

    # Third row
    plt.subplot(3, 1, 3)
    _ = plt.plot(mae_scores, marker="o", linestyle="", markersize=1.5)

    plt.title("MAE scatter of mse")

    plt.tight_layout()
    
    return mae_scores, mrse_scores, mape_scores

def windowed_residuals(y, error_metric, window_size=30):
    from numpy import array, arange
    errors_windowed = array(y)[arange(len(y)).reshape(-1, window_size)]
    errors_windowed_mean = errors_windowed.mean(axis=1)
    errors_windowed_std = errors_windowed.std(axis=1)
    n_windows = len(errors_windowed_mean)

    # errors
    plt.subplot(1, 1, 1)
    plt.errorbar(x=range(n_windows),
                 y=errors_windowed_mean,
                 yerr=errors_windowed_std,
                 fmt="-or",
                 ecolor="k",
                 markersize=4,
                 linewidth=1,
                 elinewidth=10
                )
    
    plt.xlabel("window")
    plt.ylabel(error_metric)
    plt.title(f"Mean and std of {error_metric} for {n_windows} windows")
    
    plt.tight_layout()