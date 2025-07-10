"""This is the 3W Toolkit's miscellaneous sub-module.

All resources that do not fit in the other sub-modules are define here.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import os
import configparser
import re

from matplotlib.patches import Patch
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from collections import defaultdict
from natsort import natsorted
from typing import Dict, List
from plotly.subplots import make_subplots

import warnings

warnings.simplefilter("ignore", FutureWarning)
import plotly.offline as py
import plotly.graph_objs as go
import glob
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import plotly.graph_objects as go
from typing import List, Dict

from .base import (
    COLUMNS_DATA_FILES,
    LABELS_DESCRIPTIONS,
    TRANSIENT_LABELS_DESCRIPTIONS,
    PATH_DATASET,
    TRANSIENT_OFFSET,
    VARS,
    EVENT_NAMES,
    PARQUET_EXTENSION,
    PARQUET_ENGINE,
)




# Methods
#
def label_and_file_generator(real=True, simulated=False, drawn=False):
    """This is a generating function that returns tuples for all
    indicated instance sources (`real`, `simulated` and/or
    `hand-drawn`). Each tuple refers to a specific instance and contains
    its label (int) and its full path (Path). All 3W Dataset's instances
    are considered.

    Args:
        real (bool, optional): Indicates whether `real` instances should
            be considered. Defaults to True.
        simulated (bool, optional): Indicates whether `simulated`
            instances should be considered. Defaults to False.
        drawn (bool, optional): Indicates whether `hand-drawn` instances
            should be considered. Defaults to False.

    Yields:
        generator: Tuples for all indicated instance sources. Each tuple
            refers to a specific instance and contains its label (int)
            and its full path (Path).
    """
    for i in Path(PATH_DATASET).iterdir():
        try:
            # Considers only directories
            if i.is_dir():
                label = int(i.stem)
                for fp in i.iterdir():
                    # Considers only Parquet files
                    if fp.suffix == PARQUET_EXTENSION:
                        # Considers only instances from the requested
                        # source
                        if (
                            (simulated and fp.stem.startswith("SIMULATED"))
                            or (drawn and fp.stem.startswith("DRAWN"))
                            or (
                                real
                                and (not fp.stem.startswith("SIMULATED"))
                                and (not fp.stem.startswith("DRAWN"))
                            )
                        ):
                            yield label, fp
        except:
            # Otherwise (e.g. files or directory without instances), do
            # nothing
            pass


def get_all_labels_and_files():
    """Gets lists with tuples related to all real, simulated, or
    hand-drawn instances contained in the 3w Dataset. Each list
    considers instances from a single source. Each tuple refers to a
    specific instance and contains its label (int) and its full path
    (Path).

    Returns:
        tuple: Tuple containing three lists with tuples related to real,
            simulated, and hand-drawn instances, respectively.
    """
    real_instances = list(
        label_and_file_generator(real=True, simulated=False, drawn=False)
    )
    simulated_instances = list(
        label_and_file_generator(real=False, simulated=True, drawn=False)
    )
    drawn_instances = list(
        label_and_file_generator(real=False, simulated=False, drawn=True)
    )

    return real_instances, simulated_instances, drawn_instances


def create_table_of_instances(real_instances, simulated_instances, drawn_instances):
    """Creates a table of instances (pandas.DataFrame) that shows the
    amount of instances that compose the 3W Dataset, by knowledge source
    (real, simulated and hand-drawn instances) and by instance label.

    Args:
        real_instances (list): List with tuples related to all
            real instances contained in the 3w Dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).
        simulated_instances (list): List with tuples related to all
            simulated instances contained in the 3w Dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).
        drawn_instances (list): List with tuples related to all
            hand-drawn instances contained in the 3w Dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).

    Returns:
        pandas.DataFrame: The created table that shows the amount of
            instances that compose the 3W Dataset, by knowledge source
            (real, simulated and hand-drawn instances) and by instance
            label.
    """
    # Gets the label's description of all instances as a list of dicts
    list_instances = (
        [
            {
                "INSTANCE LABEL": str(label) + " - " + LABELS_DESCRIPTIONS[label],
                "SOURCE": "REAL",
            }
            for label, fp in real_instances
        ]
        + [
            {
                "INSTANCE LABEL": str(label) + " - " + LABELS_DESCRIPTIONS[label],
                "SOURCE": "SIMULATED",
            }
            for label, fp in simulated_instances
        ]
        + [
            {
                "INSTANCE LABEL": str(label) + " - " + LABELS_DESCRIPTIONS[label],
                "SOURCE": "HAND-DRAWN",
            }
            for label, fp in drawn_instances
        ]
    )

    # Transforms the list of dicts into a pandas.DataFrame
    df_instances = pd.DataFrame(list_instances)

    # Creates the table of instances with relevant information and
    # desired format
    toi = (
        df_instances.groupby(["INSTANCE LABEL", "SOURCE"])
        .size()
        .reset_index()
        .pivot("SOURCE", "INSTANCE LABEL", 0)
        .fillna(0)
        .astype(int)
        .T
    )
    toi = toi.loc[natsorted(toi.index.values)]
    toi = toi[["REAL", "SIMULATED", "HAND-DRAWN"]]
    toi["TOTAL"] = toi.sum(axis=1)
    toi.loc["TOTAL"] = toi.sum(axis=0)

    return toi


def filter_rare_undesirable_events(toi, threshold, simulated=False, drawn=False):
    """Generates a table of instances (pandas.DataFrame) that shows the
    amount of filtered instances, by knowledge source (real, `simulated`
    and `hand-drawn` instances) and by instance label. This filter keeps
    only real instances, as well as `simulated` and `hand-drawn` if
    indicated, of rare event types. An event type is considered rare if
    the amount of instances labeled as this event relative to the total
    number of instances is less than the indicated `threshold`. In both
    totalizations, `simulated` and `hand-drawn` instances are only
    considered if indicated, but real instances are always taken into
    account.

    Args:
        toi (pandas.DataFrame): Table that shows the amount of instances
            that compose the 3W Dataset, by knowledge source (real,
            `simulated` and `hand-drawn` instances) and by instance
            label. This object is not modified in this function.
        threshold (float): Relative limit that establishes rare event
            types.
        simulated (bool, optional): Indicates whether `simulated`
            instances should be considered. Defaults to False.
        drawn (bool, optional): Indicates whether `hand-drawn` instances
            should be considered. Defaults to False.

    Returns:
        pandas.DataFrame: The table of instances (pandas.DataFrame) that
            shows the amount of filtered instances, by knowledge source
            (real, simulated and hand-drawn instances) and by instance
            label.
    """
    # Simulated and hand-drawn instances are optional, but real
    # instances are always considered
    totals = 0
    if simulated:
        totals += toi["SIMULATED"]
    if drawn:
        totals += toi["HAND-DRAWN"]
    totals += toi["REAL"]

    # Absolute limit
    limit = threshold * totals[-1]

    # Applies the filter in a new pandas.DataFrame
    rue = toi.loc[totals < limit].copy()
    rue.loc["TOTAL"] = rue.sum(axis=0)

    return rue


def load_instance(instance):
    """Loads all data and metadata from a specific `instance`.

    Args:
        instance (tuple): This tuple must refer to a specific `instance`
            and contain its label (int) and its full path (Path).

    Raises:
        Exception: Error if the Parquet file passed as arg cannot be
        read.

    Returns:
        pandas.DataFrame: Its index contains the timestamps loaded from
            the Parquet file. Its columns contain data loaded from the
            other columns of the Parquet file and metadata loaded from
            the argument `instance` (label, well, and id).
    """
    # Loads label metadata from the argument `instance`
    label, fp = instance

    try:
        # Loads well and id metadata from the argument `instance`
        well, id = fp.stem.split("_")

        # Loads data from the Parquet file
        df = pd.read_parquet(fp, engine=PARQUET_ENGINE)
        assert (
            df.columns == COLUMNS_DATA_FILES[1:]
        ).all(), f"invalid columns in the file {fp}: {df.columns.tolist()}"
    except Exception as e:
        raise Exception(f"error reading file {fp}: {e}")

    # Incorporates the loaded metadata
    df["label"] = label
    df["well"] = well
    df["id"] = id

    # Incorporates the loaded data and ordenates the df's columns
    df = df[["label", "well", "id"] + COLUMNS_DATA_FILES[1:]]

    return df


def load_instances(instances):
    """Loads all data and metadata from multiple `instances` in
    parallel.

    Args:
        instances (list): List with tuples related to real, simulated,
            or hand-drawn `instances`. Each tuple must refer to a
            specific instance and must contain its label (int) and its
            full path (Path).

    Returns:
        pandas.DataFrame: Its index contains the timestamps loaded from
            the Parquet files. Its columns contain data loaded from the
            other columns of the Parquet files and the metadata label,
            well, and id).
    """
    # Prepares for multiple parallel loadings
    pool = ThreadPool()
    dfs = []

    try:
        # Calls multiple loadings in parallel
        for df in pool.imap_unordered(load_instance, instances):
            dfs.append(df)
    finally:
        # If the instance cannot be loaded
        pool.terminate()

    # Concatenates dfs and return the result
    return pd.concat(dfs)


def create_and_plot_scatter_map(real_instances):
    """Creates and plots scatter map with all the real instances listed
    in the `real_instances` argument.

    Args:
        real_instances (list): List with tuples related to all
            real instances contained in the 3w Dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).

    Returns:
        tuple: Tuple containing the first and the last year of
            occurrence among all instances, respectively.
    """
    # Loads all instances
    df = load_instances(real_instances)

    # Finds the first and the last year of occurrence among all instances
    df_time = (
        df.reset_index()
        .groupby(["well", "id", "label"])["timestamp"]
        .agg(["min", "max"])
    )

    well_times = defaultdict(list)
    well_classes = defaultdict(list)
    for (well, id, label), (tmin, tmax) in df_time.iterrows():
        well_times[well].append((tmin, (tmax - tmin)))
        well_classes[well].append(label)

    wells = df["well"].unique()
    well_code = {w: i for i, w in enumerate(sorted(wells))}

    # Configures and plots the scatter map
    cmap = plt.get_cmap("Paired")
    my_colors = [cmap(i) for i in [3, 0, 5, 8, 11, 2, 1, 4, 9, 7, 6, 10]]
    my_cmap = mcolors.ListedColormap(my_colors, name="my_cmap")
    plt.register_cmap(name="my_cmap", cmap=my_cmap)
    cmap = plt.get_cmap("my_cmap")
    height = 5
    border = 2
    first_year = np.min(df_time["min"]).year
    last_year = np.max(df_time["max"]).year
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["font.size"] = 9
    plt.rcParams["legend.fontsize"] = 9
    fig, ax = plt.subplots(figsize=(9, 9))
    yticks = []
    yticks_labels = []
    for well in well_times.keys():
        times = well_times[well]
        class_names = well_classes[well]
        class_colors = list(map(cmap, class_names))
        well_id = well_code[well]
        yticks.append(well_id * height + height / 2 - border / 2)
        yticks_labels.append(well)
        ax.broken_barh(
            times,
            (well_id * height, height - border),
            facecolors=class_colors,
            edgecolors=class_colors,
        )
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    legend_colors = [
        Patch(facecolor=cmap(l), label=str(l) + " - " + d)
        for l, d in LABELS_DESCRIPTIONS.items()
    ]
    ax.legend(
        frameon=False,
        handles=legend_colors,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
    )

    return first_year, last_year


def count_properties_instance(instance):
    """Counts properties from a specific `instance`.

    Args:
        instance (tuple): This tuple must refer to a specific `instance`
            and contain its label (int) and its full path (Path).

    Raises:
        Exception: Error if the Parquet file passed as arg cannot be
        read.

    Returns:
        dict: Dict containing the counted properties with the following
            keys: n_vars (number of variables), n_vars_missing (number
            of missing variables), n_vars_frozen (number of frozen
            variables), n_obs (number of observations), and
            n_obs_unlabeled (number of unlabeled observations).
    """
    # Preparation for counting
    _, fp = instance
    p = {"n_vars_missing": 0, "n_vars_frozen": 0}

    try:
        # Read the Parquet file
        df = pd.read_parquet(fp, engine=PARQUET_ENGINE)
    except Exception as e:
        raise Exception(f"error reading file {fp}: {e}")

    # Counts properties
    vars = df.columns[:-1]  # Last column with class is not considered
    p["n_vars"] = len(vars)
    for var in vars:
        if df[var].isnull().all():
            p["n_vars_missing"] += 1
        u_values = df[var].unique()
        if len(u_values) == 1 and not np.isnan(u_values):
            p["n_vars_frozen"] += 1
    p["n_obs"] = len(df)
    p["n_obs_unlabeled"] = df["class"].isnull().sum()

    return p


def count_properties_instances(instances):
    """Counts properties from multiple `instances` in parallel.

    Args:
        instances (list): List with tuples related to real, simulated,
            or hand-drawn `instances`. Each tuple must refer to a
            specific instance and must contain its label (int) and its
            full path (Path).

    Returns:
        dict: Dict containing the counted properties with the following
            keys: n_vars (number of variables), n_vars_missing (number
            of missing variables), n_vars_frozen (number of frozen
            variables), n_obs (number of observations), and
            n_obs_unlabeled (number of unlabeled observations).
    """
    # Prepares for multiple parallel counts
    pool = ThreadPool()
    ps = []

    try:
        # Calls multiple counts in parallel
        for p in pool.imap_unordered(count_properties_instance, instances):
            ps.append(p)
    finally:
        # If the instance cannot be loaded
        pool.terminate()

    # Sum ps and return the result
    return dict(pd.DataFrame(ps).sum())


def calc_stats_instances(real_instances, simulated_instances, drawn_instances):
    """Calculates the 3W Dataset's fundamental aspects related to
    inherent difficulties of actual data. Three statistics are
    calculated: Missing Variables, Frozen Variables, and Unlabeled
    Observations. All instances, regardless of their source, influence
    these statistics.

    Args:
        real_instances (list): List with tuples related to all
            real instances contained in the 3w Dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).
        simulated_instances (list): List with tuples related to all
            simulated instances contained in the 3w Dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).
        drawn_instances (list): List with tuples related to all
            hand-drawn instances contained in the 3w Dataset. Each tuple
            must refer to a specific instance and must contain its label
            (int) and its full path (Path).

    Returns:
        pandas.DataFrame: Its index contains the statistic's names. Its
            columns contain statistics themselves (Amount and
            Percentage)
    """
    # Counts properties from all indicated instances
    p = count_properties_instances(
        real_instances + simulated_instances + drawn_instances
    )

    # Extract properties
    n_vars = p["n_vars"]
    n_vars_missing = p["n_vars_missing"]
    n_vars_frozen = p["n_vars_frozen"]
    n_obs = p["n_obs"]
    n_obs_unlabeled = p["n_obs_unlabeled"]

    # Calculates the statistics
    stats = {
        "Missing Variables": [
            n_vars_missing,
            f"{100*n_vars_missing/n_vars:.2f}% of {n_vars}",
        ],
        "Frozen Variables": [
            n_vars_frozen,
            f"{100*n_vars_frozen/n_vars:.2f}% of {n_vars}",
        ],
        "Unlabeled Observations": [
            n_obs_unlabeled,
            f"{100*n_obs_unlabeled/n_obs:.2f}% of {n_obs}",
        ],
    }

    return pd.DataFrame.from_dict(
        stats, orient="index", columns=["Amount", "Percentage"]
    )


def resample(data, n, class_number):
    """Downsampling for instances.

    Args:
        data (string): Instance path
        n (integer): Factor to downsampling the instance.
        class_number (integer): integer that represents the event class

    Returns:
        pandas.DataFrame: Downsamplig instance DataFrame
    """
    # Timestamp is expected to be a column
    data.reset_index(inplace=True)
    # Group Timestamp and get last value
    resampleTimestamp = data.timestamp.groupby(data.index // n).max()
    # Replace transient label from 100 to 0.5
    data["class"] = data["class"].astype(float)
    tempClassLabel = data["class"].replace(class_number + 100, 0.5)
    # Get the max value from the group Class column
    resampleClass = tempClassLabel.groupby(tempClassLabel.index // n).max()
    # Back with transient label value
    resampleClass.replace(0.5, class_number + 100, inplace=True)
    # Non overlap group and get the average value from the data
    dfResample = data.groupby(data.index // n).mean(numeric_only=True)
    # Drop class column
    dfResample.drop(["class"], axis=1, inplace=True)
    # Insert resampled class label values
    dfResample["class"] = resampleClass
    # Insert resampled timestamp
    dfResample.index = resampleTimestamp

    return dfResample


def plot_instance(class_number, instance_index, resample_factor):
    """Plot one especific event class and instance. By default the
    instance is downsampling (n=100) and Z-score Scaler. In order to
    help the visualization transient labels was changed to '0.5'.

    Args:
        class_number (integer): integer that represents the event class
        instance_index (integer): input the instance file index
    """
    instances_path = os.path.join(
        PATH_DATASET, str(class_number), "*" + PARQUET_EXTENSION
    )
    instances_path_list = glob.glob(instances_path)
    if instance_index >= len(instances_path_list):
        print(
            f"instance index {instance_index} out of range - Insert a valid index between 0 and {len(instances_path_list)-1}"
        )
    else:
        df_instance = pd.read_parquet(
            instances_path_list[instance_index], engine=PARQUET_ENGINE
        )
        df_instance_resampled = resample(df_instance, resample_factor, class_number)
        df_drop_resampled = df_instance_resampled.drop(["state", "class"], axis=1)
        df_drop_resampled.interpolate(
            method="linear", limit_direction="both", axis=0, inplace=True
        )
        df_drop_resampled.fillna(
            0,
            inplace=True,
        )
        scaler_resampled = TimeSeriesScalerMeanVariance().fit_transform(
            df_drop_resampled
        )

        df_scaler_resampled = pd.DataFrame(
            scaler_resampled.squeeze(),
            index=df_drop_resampled.index,
            columns=df_drop_resampled.columns,
        )
        df_instance_resampled["class"] = df_instance_resampled["class"].replace(
            100 + int(class_number), 0.5
        )
        df_instance_resampled["class"] = df_instance_resampled["class"].replace(
            int(class_number), 1
        )

        colors_traces = [
            "#008080",
            "#3498DB",
            "#E74C3C",
            "#884EA0",
            "#D4AC0D",
            "#AF601A",
            "#D35400",
            "#839192",
            "#2E4053",
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_scaler_resampled[VARS[0]],
                mode="lines+markers",
                marker_symbol="circle",
                marker_size=3,
                name=VARS[0],
                yaxis="y1",
                line_color=colors_traces[0],
            )
        ),
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_scaler_resampled[VARS[1]],
                mode="lines+markers",
                marker_symbol="diamond",
                marker_size=3,
                name=VARS[1],
                yaxis="y2",
                line_color=colors_traces[1],
            )
        ),
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_scaler_resampled[VARS[2]],
                mode="lines+markers",
                marker_symbol="x",
                marker_size=3,
                name=VARS[2],
                yaxis="y3",
                line_color=colors_traces[2],
            )
        ),
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_scaler_resampled[VARS[3]],
                mode="lines+markers",
                marker_symbol="star",
                marker_size=3,
                name=VARS[3],
                yaxis="y4",
                line_color=colors_traces[3],
            )
        ),
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_scaler_resampled[VARS[4]],
                mode="lines+markers",
                marker_symbol="triangle-up",
                marker_size=3,
                name=VARS[4],
                yaxis="y5",
                line_color=colors_traces[4],
            )
        ),
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_scaler_resampled[VARS[5]],
                mode="lines",
                name=VARS[5],
                yaxis="y6",
                line_color=colors_traces[5],
            )
        ),
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_scaler_resampled[VARS[6]],
                mode="lines",
                name=VARS[6],
                yaxis="y7",
                line_color=colors_traces[6],
            )
        ),
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_scaler_resampled[VARS[7]],
                mode="lines",
                name=VARS[7],
                yaxis="y8",
                line_color=colors_traces[7],
            )
        ),
        fig.add_trace(
            go.Scatter(
                x=df_instance_resampled.index,
                y=df_instance_resampled["class"],
                mode="markers",
                name="Label",
                yaxis="y9",
                line_color=colors_traces[8],
            )
        ),
        fileName = instances_path_list[instance_index].split(os.sep)
        fig.update_layout(
            title=EVENT_NAMES[class_number] + " - " + fileName[-1],
            xaxis_title="Time(s)",
            yaxis_title="z-score",
            font=dict(size=12),
            yaxis1=dict(
                tickfont=dict(color=colors_traces[0]),
                position=0,
                tickformat=".2f",
                showticklabels=False,
            ),
            yaxis2=dict(
                tickfont=dict(color=colors_traces[1]),
                overlaying="y",
                side="left",
                position=0.05,
                tickformat=".2f",
                showticklabels=False,
            ),
            yaxis3=dict(
                tickfont=dict(color=colors_traces[2]),
                overlaying="y",
                side="left",
                position=0.10,
                tickformat=".2f",
                showticklabels=False,
            ),
            yaxis4=dict(
                tickfont=dict(color=colors_traces[3]),
                overlaying="y",
                side="left",
                position=0.15,
                tickformat=".2f",
                showticklabels=False,
            ),
            yaxis5=dict(
                tickfont=dict(color=colors_traces[4]),
                overlaying="y",
                side="left",
                position=0.2,
                tickformat=".2f",
                showticklabels=False,
            ),
            yaxis6=dict(
                tickfont=dict(color=colors_traces[5]),
                overlaying="y",
                side="left",
                position=0.25,
                tickformat=".2f",
                showticklabels=False,
            ),
            yaxis7=dict(
                tickfont=dict(color=colors_traces[6]),
                overlaying="y",
                side="left",
                position=0.3,
                tickformat=".2f",
                showticklabels=False,
            ),
            yaxis8=dict(
                tickfont=dict(color=colors_traces[7]),
                overlaying="y",
                side="left",
                position=0.35,
                tickformat=".2f",
                showticklabels=False,
            ),
            yaxis9=dict(
                tickfont=dict(color=colors_traces[8]),
                anchor="x",
                overlaying="y",
                side="left",
            ),
        )
        fig.show()


class ThreeWChart:
    """A class to generate interactive visualizations for 3W dataset files using Plotly."""

    def __init__(
        self,
        file_path: str,
        title: str = "ThreeW Chart",
        y_axis: str = "P-MON-CKP",
        use_dropdown: bool = False,
        dropdown_position: tuple = (0.4, 1.4),
    ):
        """Initializes the ThreeWChart class with the given parameters.

        Args:
            file_path (str): Path to the Parquet file containing the dataset.
            title (str, optional): Title of the chart. Defaults to "ThreeW Chart".
            y_axis (str, optional): olumn name to be plotted on the y-axis. Defaults to "P-MON-CKP".
            use_dropdown (bool, optional):  Whether to show a dropdown for selecting the y-axis (default is False). Defaults to False.
            dropdown_position (tuple, optional): Position of the dropdown button on the chart. Defaults to (0.4, 1.4).
        """
        self.file_path: str = file_path
        self.title: str = title
        self.y_axis: str = y_axis
        self.use_dropdown: bool = use_dropdown
        self.dropdown_position: tuple = dropdown_position

        self.class_mapping: Dict[int, str] = self._generate_class_mapping()
        self.class_colors: Dict[int, str] = self._generate_class_colors()

    def _generate_class_mapping(self) -> Dict[int, str]:
        """Generate a combined mapping of event labels (including transient states) to their descriptions.

        Returns:
            Dict[int, str]: Mapping of event labels to their descriptions.
        """
        return {**LABELS_DESCRIPTIONS, **TRANSIENT_LABELS_DESCRIPTIONS}

    def _generate_class_colors(self) -> Dict[int, str]:
        """Generate a color mapping for event labels with specific colors applied globally.
        Colors are automatically applied based on the labels present in the data.

        Returns:
            Dict[int, str]: Mapping of event labels to their colors.
        """
        # Define global color mapping for all specific labels
        global_colors = {
            # Normal Operation - always light green
            0: "lightgreen",
            
            # Steady State labels - always red
            3: "red",
            6: "red", 
            8: "red",
            
            # Transient Condition labels - always yellow
            106: "yellow",
            108: "yellow",
            
            # State labels - specific colors
            "state_0": "darkgreen",  # Open
            "state_1": "gray",       # Shut-In
            "state_7": "magenta",    # Restart
            "state_8": "salmon",     # Depressurization
        }
        
        colors = {}

        def apply_transparency(color: str, opacity: float) -> str:
            rgb = mcolors.to_rgb(color)
            r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            return f"rgba({r}, {g}, {b}, {opacity})"

        # Apply global colors for all labels
        for label, color in global_colors.items():
            if isinstance(label, str) and label.startswith("state_"):
                # Handle state labels
                colors[label] = color
            else:
                # Handle class labels
                colors[label] = color
                # Add transient version if applicable
                if isinstance(label, int) and label > 0:
                    transient_label = label + TRANSIENT_OFFSET
                    colors[transient_label] = apply_transparency(color, opacity=0.4)
        
        # Add default colors for any other labels that might exist
        cmap = plt.get_cmap("tab10")
        for idx, (label, _) in enumerate(LABELS_DESCRIPTIONS.items()):
            if label not in colors:
                if label == 0:
                    colors[label] = "lightgreen"  # Ensure Normal Operation is always light green
                else:
                    base_color = mcolors.rgb2hex(cmap(idx % cmap.N))
                    colors[label] = base_color

                    transient_label = label + TRANSIENT_OFFSET
                    if transient_label not in colors:
                        colors[transient_label] = (
                            "lightgreen" if label == 0 else apply_transparency(base_color, opacity=0.4)
                        )
        
        return colors

    def _load_data(self) -> pd.DataFrame:
        """Loads and preprocesses the dataset using the load_instance function.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with sorted timestamps and no missing values.
        """
        instance = (int(Path(self.file_path).parent.name), Path(self.file_path))
        df = load_instance(instance)
        df.reset_index(inplace=True)
        df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").fillna(0)
        return df.sort_values(by="timestamp")

    def _get_non_zero_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns the list of columns that are not all zeros or NaN.

        Args:
            df (pd.DataFrame): DataFrame to check for non-zero columns.

        Returns:
            List[str]: List of column names that are not all zeros or NaN.
        """
        return [
            col
            for col in df.columns
            if df[col].astype(bool).sum() > 0 and col not in ["timestamp", "class"]
        ]

    def _get_background_shapes(self, df: pd.DataFrame) -> List[Dict]:
        """Creates background shapes to highlight class and state transitions in the chart.
        Creates two layers: 50% bottom for classes, 50% top for states.
        
        Special handling for Figura 7: Maps states to artificial classes in bottom layer.

        Args:
            df (pd.DataFrame): DataFrame containing the class and state data.

        Returns:
            List[Dict]: List of shape dictionaries for Plotly.
        """
        shapes = []
        
        # Check if this is Figura 7 (DRAWN_00007.parquet)
        is_figura7 = "DRAWN_00007" in str(self.file_path)
        
        if is_figura7:
            # Special handling for Figura 7: Create artificial classes based on states
            df_modified = df.copy()
            
            # Map states to artificial classes for bottom layer
            state_to_class_mapping = {
                0: 0,    # Open -> Normal Operation (lightgreen)
                1: 108,  # Shut-In -> Transient Condition (yellow)
                7: 0,    # Restart -> Normal Operation (will be overridden by state_0 in top layer)
                8: 108,  # Depressurization -> Transient Condition (yellow)
            }
            
            # Apply state-to-class mapping
            for idx, row in df_modified.iterrows():
                if 'state' in row and not pd.isna(row['state']):
                    state_val = int(row['state'])
                    if state_val in state_to_class_mapping:
                        df_modified.loc[idx, 'class'] = state_to_class_mapping[state_val]
            
            # Use modified DataFrame for class shapes
            df_for_classes = df_modified
        else:
            # Use original DataFrame for other figures
            df_for_classes = df
        
        # Track transitions for classes (bottom layer)
        prev_class = None
        start_idx = 0

        for i in range(len(df_for_classes)):
            current_class = df_for_classes.iloc[i]["class"]

            if pd.isna(current_class):
                print(f"Warning: NaN class value at index {i}")
                continue

            # Check for class transitions
            if prev_class is not None and current_class != prev_class:
                # Create shape for previous class segment (bottom 50%)
                fill_color = self.class_colors.get(prev_class, "white")
                
                shapes.append(
                    dict(
                        type="rect",
                        x0=df_for_classes.iloc[start_idx]["timestamp"],
                        x1=df_for_classes.iloc[i - 1]["timestamp"],
                        y0=0,      # Bottom layer: 0% to 50%
                        y1=0.5,
                        xref="x",
                        yref="paper",
                        fillcolor=fill_color,
                        opacity=0.2,
                        line_width=0,
                    )
                )
                start_idx = i

            prev_class = current_class

        # Add final class shape (bottom layer)
        if prev_class is not None:
            fill_color = self.class_colors.get(prev_class, "white")
                
            shapes.append(
                dict(
                    type="rect",
                    x0=df_for_classes.iloc[start_idx]["timestamp"],
                    x1=df_for_classes.iloc[len(df_for_classes) - 1]["timestamp"],
                    y0=0,      # Bottom layer: 0% to 50%
                    y1=0.5,
                    xref="x",
                    yref="paper",
                    fillcolor=fill_color,
                    opacity=0.2,
                    line_width=0,
                )
            )

        # Track transitions for states (top layer)
        if "state" in df.columns:
            prev_state = None
            start_idx = 0

            for i in range(len(df)):
                current_state = df.iloc[i]["state"]

                if pd.isna(current_state):
                    continue

                # Check for state transitions
                if prev_state is not None and current_state != prev_state:
                    # Create shape for previous state segment (top 50%)
                    if f"state_{prev_state}" in self.class_colors:
                        fill_color = self.class_colors.get(f"state_{prev_state}", "white")
                        
                        shapes.append(
                            dict(
                                type="rect",
                                x0=df.iloc[start_idx]["timestamp"],
                                x1=df.iloc[i - 1]["timestamp"],
                                y0=0.5,    # Top layer: 50% to 100%
                                y1=1,
                                xref="x",
                                yref="paper",
                                fillcolor=fill_color,
                                opacity=0.2,
                                line_width=0,
                            )
                        )
                    start_idx = i

                prev_state = current_state

            # Add final state shape (top layer)
            if prev_state is not None and f"state_{prev_state}" in self.class_colors:
                fill_color = self.class_colors.get(f"state_{prev_state}", "white")
                    
                shapes.append(
                    dict(
                        type="rect",
                        x0=df.iloc[start_idx]["timestamp"],
                        x1=df.iloc[len(df) - 1]["timestamp"],
                        y0=0.5,    # Top layer: 50% to 100%
                        y1=1,
                        xref="x",
                        yref="paper",
                        fillcolor=fill_color,
                        opacity=0.2,
                        line_width=0,
                    )
                )

        return shapes

    def _add_custom_legend(self, fig: go.Figure, present_classes: List[int], present_states: List[int] = None) -> None:
        """Adds a custom legend to the chart for classes and states present in the data.

        Args:
            fig (go.Figure): The Plotly figure to which the legend will be added.
            present_classes (List[int]): The unique class values present in the DataFrame.
            present_states (List[int], optional): The unique state values present in the DataFrame.
        """
        # Add class legends
        for class_value in present_classes:
            if class_value in self.class_mapping:
                event_name = self.class_mapping[class_value]
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=self.class_colors.get(class_value, "white"),
                            line=dict(width=1, color="black"),
                        ),
                        name=f"Class {class_value} - {event_name}",
                        showlegend=True,
                    )
                )
        
        # Add state legends if states are present
        if present_states:
            state_names = {
                0: "Open",
                1: "Shut-In", 
                7: "Restart",
                8: "Depressurization"
            }
            
            for state_value in present_states:
                state_key = f"state_{state_value}"
                if state_key in self.class_colors:
                    state_name = state_names.get(state_value, f"State {state_value}")
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(
                                size=12,
                                color=self.class_colors.get(state_key, "white"),
                                line=dict(width=1, color="black"),
                            ),
                            name=f"State {state_value} - {state_name}",
                            showlegend=True,
                        )
                    )

    def plot(self) -> None:
        """Generates and displays the interactive chart using Plotly.

        Raises:
            ValueError: If no available columns are found to plot.
        """
        df = self._load_data()

        present_classes = df["class"].dropna().unique().tolist()
        present_states = df["state"].dropna().unique().tolist() if "state" in df.columns else None

        if self.use_dropdown:
            available_y_axes = self._get_non_zero_columns(df)
            if available_y_axes:
                dropdown_buttons = [
                    dict(
                        args=[{"y": [df[col]]}, {"yaxis.title": col}],
                        label=col,
                        method="update",
                    )
                    for col in available_y_axes
                ]
                fig = go.Figure()
                if self.y_axis not in available_y_axes:
                    print(
                        f"Warning: Default y-axis '{self.y_axis}' not found in available columns."
                    )
                    print("Using the first available column as the default y-axis.")
                    self.y_axis = available_y_axes[0]
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df[self.y_axis],
                        mode="lines",
                        name="Selected Variable",
                    )
                )
                active_index = available_y_axes.index(self.y_axis)
                fig.update_layout(
                    updatemenus=[
                        dict(
                            buttons=dropdown_buttons,
                            direction="down",
                            showactive=True,
                            x=self.dropdown_position[0],
                            y=self.dropdown_position[1],
                            active=active_index,
                        )
                    ]
                )
            else:
                raise ValueError("No available columns to plot.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df[self.y_axis], mode="lines", name=self.y_axis
                )
            )

        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(
            shapes=self._get_background_shapes(df),
            xaxis_title="",
            yaxis_title=self.y_axis if not self.use_dropdown else df[self.y_axis].name,
            title=self.title,
            legend=dict(
                x=1.05, y=1, title="Legend", itemclick=False, itemdoubleclick=False
            ),
        )

        self._add_custom_legend(fig, present_classes, present_states)
        fig.show(config={"displaylogo": False})


class ThreeWChart2:
    """
    Classe para criar gráficos combinados com múltiplas escalas Y
    Combina as 4 variáveis principais em um único gráfico
    """
    
    def __init__(
        self,
        file_path: str,
        title: str = "Combined ThreeW Chart",
        variables: List[str] = None
    ):
        """
        Inicializa a classe CombinedThreeWChart
        
        Args:
            file_path (str): Caminho para o arquivo Parquet
            title (str): Título do gráfico
            variables (List[str]): Lista de variáveis a plotar
        """
        self.file_path = file_path
        self.title = title
        self.variables = variables or ["ABER-CKP", "P-MON-CKP", "P-PDG", "P-TPT"]
        
        # Configuração das cores para cada variável
        self.variable_colors = {
            'ABER-CKP': '#1f77b4',    # Azul
            'P-MON-CKP': '#ff7f0e',   # Laranja  
            'P-PDG': '#2ca02c',       # Verde
            'P-TPT': '#d62728'        # Vermelho
        }
        
        # Configuração das escalas Y
        self.yaxis_config = {
            'ABER-CKP': 'y',          # Escala esquerda externa
            'P-MON-CKP': 'y2',        # Escala esquerda interna
            'P-PDG': 'y3',            # Escala direita interna
            'P-TPT': 'y4'             # Escala direita externa
        }
        
        # Carregar unidades das variáveis
        self.variable_units = self._get_variable_units()
        
        # Carregar dados
        self.df = self._load_data()
        
        # Importar método de background shapes da classe original
        from . import ThreeWChart
        temp_chart = ThreeWChart(file_path=file_path)
        self.class_colors = temp_chart.class_colors
        self._get_background_shapes = temp_chart._get_background_shapes
    
    def _get_variable_units(self) -> Dict[str, str]:
        """Extrai as unidades das variáveis do dataset.ini"""
        
        def extract_unit_from_description(description):
            match = re.search(r'\[([^\]]+)\]', description)
            if match:
                unit = match.group(1)
                if unit == "%%": return "%"
                elif unit == "oC": return "°C"
                else: return unit
            return ""
        
        try:
            # Tentar diferentes caminhos para o dataset.ini
            possible_paths = [
                "/home/ubuntu/data_article/dataset/dataset.ini",
                "../../dataset/dataset.ini",
                "../dataset/dataset.ini"
            ]
            
            config = configparser.ConfigParser()
            units = {}
            
            for path in possible_paths:
                try:
                    config.read(path)
                    if 'PARQUET_FILE_PROPERTIES' in config:
                        for variable, description in config['PARQUET_FILE_PROPERTIES'].items():
                            if variable != 'timestamp':
                                unit = extract_unit_from_description(description)
                                units[variable.upper()] = unit
                        break
                except:
                    continue
            
            return units
        except Exception as e:
            print(f"Erro ao carregar unidades: {e}")
            return {}
    
    def _load_data(self) -> pd.DataFrame:
        """Carrega e preprocessa os dados"""
        
        instance = (int(Path(self.file_path).parent.name), Path(self.file_path))
        df = load_instance(instance)
        df.reset_index(inplace=True)
        df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").fillna(0)
        return df.sort_values(by="timestamp")
    
    def _create_yaxis_layout(self) -> Dict:
        """Cria o layout das múltiplas escalas Y"""
        
        layout = {}
        
        # Configuração das 4 escalas Y
        yaxis_configs = {
            'yaxis': {  # ABER-CKP - Esquerda externa
                'title': f"ABER-CKP ({self.variable_units.get('ABER-CKP', '')})",
                'title_font': {'color': self.variable_colors['ABER-CKP']},
                'tickfont': {'color': self.variable_colors['ABER-CKP']},
                'side': 'left',
                'position': 0
            },
            'yaxis2': {  # P-MON-CKP - Esquerda interna
                'title': f"P-MON-CKP ({self.variable_units.get('P-MON-CKP', '')})",
                'title_font': {'color': self.variable_colors['P-MON-CKP']},
                'tickfont': {'color': self.variable_colors['P-MON-CKP']},
                'side': 'left',
                'overlaying': 'y',
                'position': 0.1
            },
            'yaxis3': {  # P-PDG - Direita interna
                'title': f"P-PDG ({self.variable_units.get('P-PDG', '')})",
                'title_font': {'color': self.variable_colors['P-PDG']},
                'tickfont': {'color': self.variable_colors['P-PDG']},
                'side': 'right',
                'overlaying': 'y',
                'position': 0.9
            },
            'yaxis4': {  # P-TPT - Direita externa
                'title': f"P-TPT ({self.variable_units.get('P-TPT', '')})",
                'title_font': {'color': self.variable_colors['P-TPT']},
                'tickfont': {'color': self.variable_colors['P-TPT']},
                'side': 'right',
                'overlaying': 'y',
                'position': 1
            }
        }
        
        layout.update(yaxis_configs)
        return layout
    
    def _add_variable_trace(self, fig: go.Figure, variable: str) -> None:
        """Adiciona uma trace para uma variável específica"""
        
        if variable not in self.df.columns:
            print(f"Variável {variable} não encontrada nos dados")
            return
        
        yaxis = self.yaxis_config.get(variable, 'y')
        color = self.variable_colors.get(variable, '#000000')
        
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df[variable],
                mode='lines',
                name=variable,
                line=dict(color=color, width=2),
                yaxis=yaxis,
                showlegend=True
            )
        )
    
    def _add_background_shapes(self, fig: go.Figure) -> None:
        """Adiciona as shapes de fundo (classes e estados)"""
        
        try:
            shapes = self._get_background_shapes(self.df)
            fig.update_layout(shapes=shapes)
        except Exception as e:
            print(f"Erro ao adicionar background shapes: {e}")
    
    def plot(self) -> None:
        """Gera e exibe o gráfico combinado"""
        
        # Criar figura base
        fig = go.Figure()
        
        # Adicionar traces para cada variável
        for variable in self.variables:
            self._add_variable_trace(fig, variable)
        
        # Configurar layout com múltiplas escalas Y
        yaxis_layout = self._create_yaxis_layout()
        
        fig.update_layout(
            title=self.title,
            xaxis_title="",  # Sem título no eixo X
            **yaxis_layout,
            legend=dict(
                x=1.15, 
                y=1, 
                title="Variáveis",
                itemclick=False, 
                itemdoubleclick=False
            ),
            margin=dict(l=100, r=150, t=50, b=50),  # Margens para acomodar escalas
            width=1200,
            height=600
        )
        
        # Adicionar background shapes (classes e estados)
        self._add_background_shapes(fig)
        
        # Remover barra de navegação
        fig.update_xaxes(rangeslider_visible=False)
        
        # Exibir gráfico
        fig.show(config={"displaylogo": False})

class ThreeWChart3:
    """
    Classe para criar gráficos combinados com múltiplas escalas Y
    Combina as 4 variáveis principais em um único gráfico
    """
    
    def __init__(
        self,
        file_path: str,
        title: str = "Combined ThreeW Chart",
        variables: List[str] = None
    ):
        """
        Inicializa a classe CombinedThreeWChart
        
        Args:
            file_path (str): Caminho para o arquivo Parquet
            title (str): Título do gráfico
            variables (List[str]): Lista de variáveis a plotar
        """
        self.file_path = file_path
        self.title = title
        self.variables = variables or ["ABER-CKP", "P-MON-CKP", "P-PDG", "P-TPT"]
        
        # Configuração das cores para cada variável (linhas)
        self.variable_colors = {
            'ABER-CKP': '#ff7f0e',    # Laranja
            'P-MON-CKP': '#1f77b4',   # Azul
            'P-PDG': '#d62728',       # Vermelho
            'P-TPT': '#2ca02c'        # Verde
        }
        
        # Configuração das cores das escalas
        self.scale_colors = {
            '%': '#ff7f0e',    # Laranja para escala de % (esquerda)
            'Pa': '#000000'    # Preto para escala de Pa (direita)
        }
        
        # Configuração das escalas Y (apenas 2 escalas)
        self.yaxis_config = {
            'ABER-CKP': 'y',          # Escala esquerda (%)
            'P-MON-CKP': 'y2',        # Escala direita (Pa)
            'P-PDG': 'y2',            # Escala direita (Pa)
            'P-TPT': 'y2'             # Escala direita (Pa)
        }
        
        # Carregar unidades das variáveis
        self.variable_units = self._get_variable_units()
        
        # Carregar dados
        self.df = self._load_data()
        
        # Importar método de background shapes da classe original
        from . import ThreeWChart
        temp_chart = ThreeWChart(file_path=file_path)
        self.class_colors = temp_chart.class_colors
        self._get_background_shapes = temp_chart._get_background_shapes
    
    def _get_variable_units(self) -> Dict[str, str]:
        """Extrai as unidades das variáveis do dataset.ini"""
        
        def extract_unit_from_description(description):
            match = re.search(r'\[([^\]]+)\]', description)
            if match:
                unit = match.group(1)
                if unit == "%%": return "%"
                elif unit == "oC": return "°C"
                else: return unit
            return ""
        
        try:
            # Tentar diferentes caminhos para o dataset.ini
            possible_paths = [
                "/home/ubuntu/data_article/dataset/dataset.ini",
                "../../dataset/dataset.ini",
                "../dataset/dataset.ini"
            ]
            
            config = configparser.ConfigParser()
            units = {}
            
            for path in possible_paths:
                try:
                    config.read(path)
                    if 'PARQUET_FILE_PROPERTIES' in config:
                        for variable, description in config['PARQUET_FILE_PROPERTIES'].items():
                            if variable != 'timestamp':
                                unit = extract_unit_from_description(description)
                                units[variable.upper()] = unit
                        break
                except:
                    continue
            
            return units
        except Exception as e:
            print(f"Erro ao carregar unidades: {e}")
            return {}
    
    def _load_data(self) -> pd.DataFrame:
        """Carrega e preprocessa os dados"""
        
        instance = (int(Path(self.file_path).parent.name), Path(self.file_path))
        df = load_instance(instance)
        df.reset_index(inplace=True)
        df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").fillna(0)
        return df.sort_values(by="timestamp")
    
    def _create_yaxis_layout(self) -> Dict:
        """Cria o layout com apenas 2 escalas Y (% e Pa)"""
        
        layout = {}
        
        # Configuração das 2 escalas Y
        yaxis_configs = {
            'yaxis': {  # Escala esquerda para % (ABER-CKP)
                'title': f"Porcentagem (%)",
                'title_font': {'color': self.scale_colors['%']},
                'tickfont': {'color': self.scale_colors['%']},
                'side': 'left'
            },
            'yaxis2': {  # Escala direita para Pa (P-MON-CKP, P-PDG, P-TPT)
                'title': f"Pressão (Pa)",
                'title_font': {'color': self.scale_colors['Pa']},
                'tickfont': {'color': self.scale_colors['Pa']},
                'side': 'right',
                'overlaying': 'y'
            }
        }
        
        layout.update(yaxis_configs)
        return layout
    
    def _add_variable_trace(self, fig: go.Figure, variable: str) -> None:
        """Adiciona uma trace para uma variável específica"""
        
        if variable not in self.df.columns:
            print(f"Variável {variable} não encontrada nos dados")
            return
        
        yaxis = self.yaxis_config.get(variable, 'y')
        color = self.variable_colors.get(variable, '#000000')
        
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df[variable],
                mode='lines',
                name=variable,
                line=dict(color=color, width=2),
                yaxis=yaxis,
                showlegend=True
            )
        )
    
    def _add_background_shapes(self, fig: go.Figure) -> None:
        """Adiciona as shapes de fundo (classes e estados)"""
        
        try:
            shapes = self._get_background_shapes(self.df)
            fig.update_layout(shapes=shapes)
        except Exception as e:
            print(f"Erro ao adicionar background shapes: {e}")
    
    def plot(self) -> None:
        """Gera e exibe o gráfico combinado"""
        
        # Criar figura base
        fig = go.Figure()
        
        # Adicionar traces para cada variável
        for variable in self.variables:
            self._add_variable_trace(fig, variable)
        
        # Configurar layout com múltiplas escalas Y
        yaxis_layout = self._create_yaxis_layout()
        
        fig.update_layout(
            title=self.title,
            xaxis_title="",  # Sem título no eixo X
            **yaxis_layout,
            legend=dict(
                x=1.15, 
                y=1, 
                title="Variáveis",
                itemclick=False, 
                itemdoubleclick=False
            ),
            margin=dict(l=100, r=150, t=50, b=50),  # Margens para acomodar escalas
            width=1200,
            height=600
        )
        
        # Adicionar background shapes (classes e estados)
        self._add_background_shapes(fig)
        
        # Remover barra de navegação
        fig.update_xaxes(rangeslider_visible=False)
        
        # Exibir gráfico
        fig.show(config={"displaylogo": False})
