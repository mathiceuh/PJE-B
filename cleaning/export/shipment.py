# ===============================================================
#  shipment.py — Step 4: Shipping the cleaned results
# ===============================================================

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import json


# ===============================================================
#  Column Selection Helper
# ===============================================================
def select_columns(df: pd.DataFrame, keep_extra=True, label_idx=None, tweet_idx=None) -> pd.DataFrame:
    """
    Return a DataFrame subset according to keep_extra settings.
    """
    if keep_extra is True:
        return df.copy()

    elif keep_extra is False:
        # ✅ Prefer explicitly passed indices if available
        if label_idx is not None and tweet_idx is not None:
            return df.iloc[:, [label_idx, tweet_idx]].copy()
        else:
            # fallback: first two columns
            return df.iloc[:, :2].copy()



# ===============================================================
#  Base Class
# ===============================================================
class BaseShipper(ABC):
    """
    Abstract base class for all shipment implementations.
    Each subclass defines how to 'ship' (export or return) cleaned data.
    """

    @abstractmethod
    def ship(self, df: pd.DataFrame, keep_extra=True):
        """
        Ship or return the cleaned DataFrame.
        Parameters
        ----------
        df : pandas.DataFrame
            The cleaned dataset.
        keep_extra : bool | list[int] | list[str]
            Columns to keep (see select_columns()).
        """
        pass


# ===============================================================
#  CSV Shipper
# ===============================================================
class CSVShipper(BaseShipper):
    """Exports the cleaned dataset to a CSV file."""

    def __init__(self, output_path="cleaned_tweets.csv"):
        self.output_path = Path(output_path)

    def ship(self, df: pd.DataFrame, keep_extra=True, label_idx=None, tweet_idx=None):
        df_out = select_columns(df, keep_extra, label_idx=label_idx, tweet_idx=tweet_idx)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(self.output_path, index=False, encoding="utf-8")
        print(f"✅ CSV shipped → {self.output_path.resolve()}")
        return self.output_path


# ===============================================================
#  JSON Shipper
# ===============================================================
class JSONShipper(BaseShipper):
    """Exports the cleaned dataset to a JSON file."""

    def __init__(self, output_path="cleaned_tweets.json"):
        self.output_path = Path(output_path)

    def ship(self, df: pd.DataFrame, keep_extra=True, label_idx=None, tweet_idx=None):
        df_out = select_columns(df, keep_extra, label_idx=label_idx, tweet_idx=tweet_idx)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_json(self.output_path, orient="records", force_ascii=False, indent=2)
        print(f"✅ JSON shipped → {self.output_path.resolve()}")
        return self.output_path


# ===============================================================
#  DataFrame Shipper
# ===============================================================
class DataFrameShipper(BaseShipper):
    """Returns a cleaned pandas DataFrame (useful for GUI or in-memory pipelines)."""

    def ship(self, df: pd.DataFrame, keep_extra=True, label_idx=None, tweet_idx=None):
        df_out = select_columns(df, keep_extra, label_idx=label_idx, tweet_idx=tweet_idx)
        print(f"✅ DataFrame ready → {df_out.shape[0]} rows, {df_out.shape[1]} columns")
        return df_out


# ===============================================================
#  Custom Object Shipper
# ===============================================================
@dataclass
class TweetRecord:
    """Lightweight Python representation of a cleaned tweet."""
    label: str
    tweet: str
    extras: dict | None = None


class ObjectShipper(BaseShipper):
    """Returns a list of TweetRecord objects (for GUI or in-memory use)."""

    def ship(self, df: pd.DataFrame, keep_extra=True):
        df_out = select_columns(df, keep_extra)

        if df_out.shape[1] < 2:
            raise ValueError("The dataset must have at least two columns (label + tweet).")

        records = []
        for _, row in df_out.iterrows():
            label = row.iloc[0]
            tweet = row.iloc[1]

            extras = None
            if keep_extra is True and len(df_out.columns) > 2:
                extras = {col: row[col] for col in df_out.columns[2:]}
            elif isinstance(keep_extra, (list, tuple)) and len(df_out.columns) > 2:
                extras = {col: row[col] for col in df_out.columns[2:]}

            records.append(TweetRecord(label=label, tweet=tweet, extras=extras))

        print(f"✅ Created {len(records)} TweetRecord objects.")
        return records


class ShipmentManager:
    """
    Factory/dispatcher for selecting a shipper type dynamically.
    """

    def __init__(self, mode="csv", **kwargs):
        self.mode = mode.lower()
        self.kwargs = kwargs

    def ship(self, df: pd.DataFrame, keep_extra=True, **extra_args):
        if self.mode == "csv":
            return CSVShipper(**self.kwargs).ship(df, keep_extra, **extra_args)
        elif self.mode == "json":
            return JSONShipper(**self.kwargs).ship(df, keep_extra, **extra_args)
        elif self.mode == "object":
            return ObjectShipper().ship(df, keep_extra, **extra_args)
        elif self.mode == "dataframe":
            return DataFrameShipper().ship(df, keep_extra, **extra_args)
        else:
            raise ValueError(f"❌ Unknown shipment mode: {self.mode}")


