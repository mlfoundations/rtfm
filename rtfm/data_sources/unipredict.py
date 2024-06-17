import pandas as pd


def parse_unipredict_dataset_name(name: str) -> str:
    if name == "team-ai-spam-text-message-classification":
        return "team-ai/spam-text-message-classification"
    else:
        return name.replace("-", "/", 1)


def format_target_column(column: pd.Series, num_buckets: int = 4) -> pd.Series:
    assert pd.api.types.is_numeric_dtype(column)
    assert num_buckets > 0, "Number of buckets must be greater than zero."

    # Compute bucket thresholds
    thresholds = [column.quantile(i / num_buckets) for i in range(1, num_buckets)]

    # Define a function to categorize each value
    def categorize_value(x):
        for i, threshold in enumerate(thresholds):
            if x < threshold:
                if i == 0:
                    return f"less than {threshold}"
                else:
                    return f"between {thresholds[i-1]} and {threshold}"
        return f"greater than {thresholds[-1]}"

    # Apply the categorization function to the column
    return column.apply(categorize_value)
