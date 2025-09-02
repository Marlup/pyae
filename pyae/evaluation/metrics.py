from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt

from torchmetrics import(
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
    ExplainedVariance,
    R2Score
)

from numpy import (
    mean, 
    std, 
    linspace, 
    array, 
    arange,
    zeros_like, 
    diag, 
    eye
)

def get_confusion_matrix(y_true, y_pred):
    cf = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    return cf, disp

def transform_prob_prediction(y_pred_prob, k=1, hard=True):
    pred_np = y_pred_prob.numpy() if isinstance(y_pred_prob, torch.Tensor) else y_pred_prob
    if hard:
        return pred_np.argmax(axis=1)
    return pred_np.argsort(axis=1)[:, -k:][:, ::-1]

def k_neighbors_adjusted_matrix(cf):
    n = len(cf)
    adjusted = cf.copy()
    for i in range(n):
        if i > 0:
            adjusted[i, i] += cf[i, i-1]
        if i < n-1:
            adjusted[i, i] += cf[i, i+1]
        adjusted[i, i-1 if i > 0 else i] = 0
        adjusted[i, i+1 if i < n-1 else i] = 0
    return adjusted

def k_neighbors_confusion_matrix(y_test, y_pred):
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