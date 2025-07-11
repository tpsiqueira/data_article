import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import configparser
import re
import warnings
import plotly.graph_objects as go

from pathlib import Path
from typing import Dict, List

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

warnings.simplefilter("ignore", FutureWarning)



# Methods
#



def load_instance(instance):
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


class ThreeWChart:
    
    def __init__(
        self,
        file_path: str,
        title: str = "ThreeW Chart",
        y_axis: str = "P-MON-CKP",
        use_dropdown: bool = False,
        dropdown_position: tuple = (0.4, 1.4),
    ):      
        self.file_path: str = file_path
        self.title: str = title
        self.y_axis: str = y_axis
        self.use_dropdown: bool = use_dropdown
        self.dropdown_position: tuple = dropdown_position

        self.class_mapping: Dict[int, str] = self._generate_class_mapping()
        self.class_colors: Dict[int, str] = self._generate_class_colors()

    def _generate_class_mapping(self) -> Dict[int, str]:      
        return {**LABELS_DESCRIPTIONS, **TRANSIENT_LABELS_DESCRIPTIONS}

    def _generate_class_colors(self) -> Dict[int, str]:              
        global_colors = {
            # Normal Operation -  light green
            0: "lightgreen",
            
            # Steady State labels -  red
            2: "red",
            3: "red",
            6: "red", 
            8: "red",
            
            # Transient Condition labels - yellow
            102: "yellow",
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
        
        for label, color in global_colors.items():
            if isinstance(label, str) and label.startswith("state_"):                
                colors[label] = color
            else:                
                colors[label] = color                
                if isinstance(label, int) and label > 0:
                    transient_label = label + TRANSIENT_OFFSET
                    colors[transient_label] = apply_transparency(color, opacity=0.4)
                
        cmap = plt.get_cmap("tab10")
        for idx, (label, _) in enumerate(LABELS_DESCRIPTIONS.items()):
            if label not in colors:
                if label == 0:
                    colors[label] = "lightgreen"
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
        instance = (int(Path(self.file_path).parent.name), Path(self.file_path))
        df = load_instance(instance)
        df.reset_index(inplace=True)
        df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").fillna(0)
        return df.sort_values(by="timestamp")

    def _get_non_zero_columns(self, df: pd.DataFrame) -> List[str]:       
        return [
            col
            for col in df.columns
            if df[col].astype(bool).sum() > 0 and col not in ["timestamp", "class"]
        ]

    def _get_background_shapes(self, df: pd.DataFrame) -> List[Dict]:  
        shapes = []
                
        is_figura7 = "DRAWN_00007" in str(self.file_path)
        
        if is_figura7:            
            df_modified = df.copy()
                        
            state_to_class_mapping = {
                0: 0,    # Open -> Normal Operation (lightgreen)
                1: 108,  # Shut-In -> Transient Condition (yellow)
                7: 0,    # Restart -> Normal Operation (darkgreen)
                8: 108,  # Depressurization -> Transient Condition (yellow)
            }
                        
            for idx, row in df_modified.iterrows():
                if 'state' in row and not pd.isna(row['state']):
                    state_val = int(row['state'])
                    if state_val in state_to_class_mapping:
                        df_modified.loc[idx, 'class'] = state_to_class_mapping[state_val]
                        
            df_for_classes = df_modified
        else:            
            df_for_classes = df
                
        prev_class = None
        start_idx = 0

        for i in range(len(df_for_classes)):
            current_class = df_for_classes.iloc[i]["class"]

            if pd.isna(current_class):
                print(f"Warning: NaN class value at index {i}")
                continue
            
            if prev_class is not None and current_class != prev_class:                
                fill_color = self.class_colors.get(prev_class, "white")
                
                shapes.append(
                    dict(
                        type="rect",
                        x0=df_for_classes.iloc[start_idx]["timestamp"],
                        x1=df_for_classes.iloc[i - 1]["timestamp"],
                        y0=0,
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
        
        if prev_class is not None:
            fill_color = self.class_colors.get(prev_class, "white")
                
            shapes.append(
                dict(
                    type="rect",
                    x0=df_for_classes.iloc[start_idx]["timestamp"],
                    x1=df_for_classes.iloc[len(df_for_classes) - 1]["timestamp"],
                    y0=0,
                    y1=0.5,
                    xref="x",
                    yref="paper",
                    fillcolor=fill_color,
                    opacity=0.2,
                    line_width=0,
                )
            )
        
        if "state" in df.columns:
            prev_state = None
            start_idx = 0

            for i in range(len(df)):
                current_state = df.iloc[i]["state"]

                if pd.isna(current_state):
                    continue
                
                if prev_state is not None and current_state != prev_state:                    
                    if f"state_{prev_state}" in self.class_colors:
                        fill_color = self.class_colors.get(f"state_{prev_state}", "white")
                        
                        shapes.append(
                            dict(
                                type="rect",
                                x0=df.iloc[start_idx]["timestamp"],
                                x1=df.iloc[i - 1]["timestamp"],
                                y0=0.5,
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
            
            if prev_state is not None and f"state_{prev_state}" in self.class_colors:
                fill_color = self.class_colors.get(f"state_{prev_state}", "white")
                    
                shapes.append(
                    dict(
                        type="rect",
                        x0=df.iloc[start_idx]["timestamp"],
                        x1=df.iloc[len(df) - 1]["timestamp"],
                        y0=0.5,
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
    
    def __init__(
        self,
        file_path: str,
        title: str = "Combined ThreeW Chart",
        variables: List[str] = None
    ):       
        self.file_path = file_path
        self.title = title
        self.variables = variables or ["ABER-CKP", "P-MON-CKP", "P-PDG", "P-TPT"]
                
        self.variable_colors = {
            'ABER-CKP': '#1f77b4',    # Azul
            'P-MON-CKP': '#ff7f0e',   # Laranja  
            'P-PDG': '#2ca02c',       # Verde
            'P-TPT': '#d62728'        # Vermelho
        }
        
        # Configuração das escalas Y
        self.yaxis_config = {
            'ABER-CKP': 'y',
            'P-MON-CKP': 'y2',
            'P-PDG': 'y3',
            'P-TPT': 'y4'
        }
                
        self.variable_units = self._get_variable_units()
                
        self.df = self._load_data()
                
        from . import ThreeWChart
        temp_chart = ThreeWChart(file_path=file_path)
        self.class_colors = temp_chart.class_colors
        self._get_background_shapes = temp_chart._get_background_shapes
    
    def _get_variable_units(self) -> Dict[str, str]:        
        
        def extract_unit_from_description(description):
            match = re.search(r'\[([^\]]+)\]', description)
            if match:
                unit = match.group(1)
                if unit == "%%": return "%"
                elif unit == "oC": return "°C"
                else: return unit
            return ""
        
        try:            
            possible_paths = [                
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
        
        instance = (int(Path(self.file_path).parent.name), Path(self.file_path))
        df = load_instance(instance)
        df.reset_index(inplace=True)
        df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").fillna(0)
        return df.sort_values(by="timestamp")
    
    def _create_yaxis_layout(self) -> Dict:        
        
        layout = {}
                
        yaxis_configs = {
            'yaxis': {
                'title': f"ABER-CKP ({self.variable_units.get('ABER-CKP', '')})",
                'title_font': {'color': self.variable_colors['ABER-CKP']},
                'tickfont': {'color': self.variable_colors['ABER-CKP']},
                'side': 'left',
                'position': 0
            },
            'yaxis2': {
                'title': f"P-MON-CKP ({self.variable_units.get('P-MON-CKP', '')})",
                'title_font': {'color': self.variable_colors['P-MON-CKP']},
                'tickfont': {'color': self.variable_colors['P-MON-CKP']},
                'side': 'left',
                'overlaying': 'y',
                'position': 0.1
            },
            'yaxis3': {
                'title': f"P-PDG ({self.variable_units.get('P-PDG', '')})",
                'title_font': {'color': self.variable_colors['P-PDG']},
                'tickfont': {'color': self.variable_colors['P-PDG']},
                'side': 'right',
                'overlaying': 'y',
                'position': 0.9
            },
            'yaxis4': {
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
        
        try:
            shapes = self._get_background_shapes(self.df)
            fig.update_layout(shapes=shapes)
        except Exception as e:
            print(f"Erro ao adicionar background shapes: {e}")
    
    def plot(self) -> None:        
                
        fig = go.Figure()
                
        for variable in self.variables:
            self._add_variable_trace(fig, variable)
                
        yaxis_layout = self._create_yaxis_layout()
        
        fig.update_layout(
            title=self.title,
            xaxis_title="",
            **yaxis_layout,
            legend=dict(
                x=1.15, 
                y=1, 
                title="Legend",
                itemclick=False, 
                itemdoubleclick=False
            ),
            margin=dict(l=100, r=150, t=50, b=50),
            width=1200,
            height=600
        )
                
        self._add_background_shapes(fig)
                
        fig.update_xaxes(rangeslider_visible=False)
                
        fig.show(config={"displaylogo": False})

