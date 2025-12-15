from annotator import KeywordAnnotator
from evaluate import evaluate_accuracy  # or paste function here

# 1️⃣ Annotate the dataset
annotator = KeywordAnnotator(json_path="keywords.json")
df = annotator.annotate("test.csv", "output.csv", mode="add")

# 2️⃣ Evaluate accuracy
evaluate_accuracy(df, true_col="0", pred_col="predicted_label")
