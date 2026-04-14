import pandas as pd
from src.configuration import (
    MADRID_FILE,
    SEGOVIA_FILE,
    PRE_POST_METRICS,
    MADRID_SEMINAR,
    SEGOVIA_SEMINAR,
)

def load_site_data(path, site_name, drop_teachers=False):
    """Load data for a single site, optionally dropping teachers."""
    df = pd.read_excel(path).copy()
    df["site"] = site_name

    if drop_teachers:
        df = df[df["Relationship to IE"] != "Professor/Lecturer"].copy()

    return df


def load_all_data():
    """Load and combine data from both sites, dropping teachers and adding seminar attendance information."""
    madrid = load_site_data(MADRID_FILE, "Madrid", drop_teachers=True)
    segovia = load_site_data(SEGOVIA_FILE, "Segovia", drop_teachers=True)

    madrid["seminar_attendee"] = madrid["Participant"].isin(MADRID_SEMINAR)
    segovia["seminar_attendee"] = segovia["Participant"].isin(SEGOVIA_SEMINAR)

    return madrid, segovia, pd.concat([madrid, segovia], ignore_index=True)


def build_long_prepost(df):
    """Reshape the data to a long format for pre/post metrics.  
    The resulting DataFrame will have columns: Participant, site, metric, time (Before/After), and value."""
    rows = []

    for metric_name, (pre_col, post_col) in PRE_POST_METRICS.items():
        temp_pre = df[["Participant", "site"]].copy()
        temp_pre["metric"] = metric_name
        temp_pre["time"] = "Before"
        temp_pre["value"] = pd.to_numeric(df[pre_col], errors="coerce")

        temp_post = df[["Participant", "site"]].copy()
        temp_post["metric"] = metric_name
        temp_post["time"] = "After"
        temp_post["value"] = pd.to_numeric(df[post_col], errors="coerce")

        rows.append(temp_pre)
        rows.append(temp_post)

    return pd.concat(rows, ignore_index=True)


def build_change_scores(df):
    """Calculate change scores for each pre/post metric.
    The resulting DataFrame will have columns: Participant, site, seminar_attendee, and change scores for each metric."""
    
    out = df[["Participant", "site", "seminar_attendee"]].copy()

    for metric_name, (pre_col, post_col) in PRE_POST_METRICS.items():
        out[f"{metric_name}_before"] = pd.to_numeric(df[pre_col], errors="coerce")
        out[f"{metric_name}_after"] = pd.to_numeric(df[post_col], errors="coerce")
        out[f"{metric_name}_change"] = out[f"{metric_name}_after"] - out[f"{metric_name}_before"]

    return out

