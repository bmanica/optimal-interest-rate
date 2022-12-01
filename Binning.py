# -- Binning categories for scorecard alike representation -- #

# Libraries to use
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union

bins = {
    "customer_age": [
        {
            "label": "(-inf, 37)",
            "max": 37
        },
        {
            "label": "(37, 40)",
            "max": 40
        },
        {
            "label": "(40, 46)",
            "max": 46
        },
        {
            "label": "(46, 58)",
            "max": 58
        },
        {
            "label": "(58, inf)",
            "max": sys.maxsize
        },
    ],
    "income": [
        {
            "label": "(-inf, 29000)",
            "max": 29000
        },
        {
            "label": "(29000, 33000)",
            "max": 33000
        },
        {
            "label": "(33000, 42000)",
            "max": 42000
        },
        {
            "label": "(42000, 47000)",
            "max": 47000
        },
        {
            "label": "(47000, inf)",
            "max": sys.maxsize
        }
    ],
    "months_with_bank": [
        {
            "label": "(-inf, 12)",
            "max": 12
        },
        {
            "label": "(12, 45)",
            "max": 45
        },
        {
            "label": "(45, 71)",
            "max": 71
        },
        {
            "label": "(71, inf)",
            "max": sys.maxsize
        },
    ],
    "months_at_address": [
        {
            "label": "(-inf, 23)",
            "max": 23
        },
        {
            "label": "(23, 83)",
            "max": 83
        },
        {
            "label": "(83, inf)",
            "max": sys.maxsize
        },
    ],
    "balance": [
        {
            "label": "(-inf, 558)",
            "max": 558
        },
        {
            "label": "(558, 1254)",
            "max": 1254
        },
        {
            "label": "(1254, 1597)",
            "max": 1597
        },
        {
            "label": "(1597, inf)",
            "max": sys.maxsize
        },
    ],
}

def get_label(val: float, var: str):
    config = bins[var]
    for bin_info in bins[var]:
        if val <= bin_info["max"]:
            return bin_info["label"]
    return bins[var][-1]["label"]

def get_absolute_odds(
    df: pd.DataFrame,
    col: str,
    target: str = "status",
    target_mappings: Dict = {0: "good", 1: "bad"},
    absolute_values: bool = False
):
    # Configuration of the target map
    key_first, key_second =  list(target_mappings.keys())
    # Calculate absolute odds & assign good/bad labels
    return (
        df.query(f"{target}=={key_first}")
        .groupby(col).size().reset_index()
        .rename(columns={0: target_mappings[key_first]})
        .set_index(col)
    ).join(
        df.query(f"{target}=={key_second}")
        .groupby(col).size().reset_index()
        .rename(columns={0: target_mappings[key_second]})
        .set_index(col)
    ).reset_index()[[col, "good", "bad"]]


def caclulate_relative_odds(row: pd.Series, total_good: int, total_bad: int) -> pd.Series:
    return pd.Series(
        {
            **row.to_dict(),
            "good": row["good"] / total_good,
            "bad": row["bad"] / total_bad
        }
    )

def get_odds(
    df: pd.DataFrame,
    col: str,
    target: str = "status",
    target_mappings: Dict = {0: "good", 1: "bad"},
    absolute_values: bool = False
):
    # Configuration of the target map
    key_first, key_second =  list(target_mappings.keys())
    # Calculate absolute odds & assign good/bad labels
    odds_absolute = get_absolute_odds(df, col, target, target_mappings, absolute_values)
    if absolute_values:
        return odds_absolute
    # Calculate relative odds
    total_good = odds_absolute["good"].sum()
    total_bad = odds_absolute["bad"].sum()
    return odds_absolute.apply(
        lambda row: caclulate_relative_odds(
            row=row,
            total_good=total_good,
            total_bad=total_bad
        ),
        axis=1
    )

def calculate_woe(row: pd.Series) -> pd.Series:
    return pd.Series(
        {
            **row.to_dict(),
            "woe": np.log(row["good"] / row["bad"]),
            "info_val": (row["good"] - row["bad"]) * np.log(row["good"] / row["bad"])
        }
    )

def get_woe(
    df: pd.DataFrame,
    col: str,
    target: str = "status",
    target_mappings: Dict = {0: "good", 1: "bad"}
) -> pd.DataFrame:
    return get_odds(df, col, target, target_mappings, absolute_values=False)\
        .apply(lambda row: calculate_woe(row), axis=1)\
        .sort_values(by="woe", axis=0, ascending=True)
