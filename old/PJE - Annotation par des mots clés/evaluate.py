import pandas as pd

def evaluate_accuracy(df: pd.DataFrame, true_col: str = "label", pred_col: str = "predicted_label") -> float:
    """
    Compare predicted vs. true labels and print accuracy.
    """
    # Ensure both columns exist
    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Missing columns '{true_col}' or '{pred_col}' in DataFrame.")

    # Drop rows with NaN labels
    df = df.dropna(subset=[true_col, pred_col])

    # Convert to int
    y_true = df[true_col].astype(int)
    y_pred = df[pred_col].astype(int)

    # Compute accuracy
    correct = (y_true == y_pred).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0.0

    print(f"✅ Accuracy: {accuracy * 100:.2f}%  ({correct}/{total} correct)")
    return accuracy
