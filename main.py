from typing import cast, Literal
from functools import cached_property
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


class Era5TemperatureReader:
    """Load quarterly ERA5 temperature files and concatenate over time."""

    QUARTER_RE: re.Pattern = re.compile(r"^\d{4}q[1-4]$")

    def __init__(self, root_dir: str | Path, var_name: Literal["skt", "t2m"]) -> None:
        self.root_dir = Path(root_dir)
        self.var_name = var_name
        self.file_re: re.Pattern
        if var_name == "skt":
            self.file_re = re.compile(r"^skin_temperature.*\.nc$")
        else:
            self.file_re = re.compile(r"^2m_temperature.*\.nc$")

    @cached_property
    def filepaths(self) -> list[Path]:
        quarter_dirs: list[Path] = sorted(
            p for p in self.root_dir.iterdir()
            if p.is_dir() and self.QUARTER_RE.match(p.name)
        )
        filepaths: list[Path] = []
        for quarter_dir in quarter_dirs:
            matches: list[Path] = sorted(quarter_dir.glob(self.file_re.pattern))
            assert len(matches) == 1, f"{quarter_dir}: expected 1 match, got {len(matches)}"
            filepaths.append(matches[0])

        return filepaths

    @cached_property
    def dataset(self) -> xr.Dataset:
        datasets: list[xr.Dataset] = [xr.open_dataset(path) for path in self.filepaths]
        combined: xr.Dataset = xr.concat(datasets, dim="valid_time").sortby("valid_time")
        return combined

    @cached_property
    def dataarray(self) -> xr.DataArray:
        return self.dataset[self.var_name]

    @cached_property
    def daily_mean(self) -> pd.Series:
        """ Compute daily spatial mean of temperature"""
        mean: xr.DataArray = self.dataarray.mean(dim=("latitude", "longitude"))
        mean_series: pd.Series = cast(pd.Series, mean.to_pandas())
        assert isinstance(mean_series, pd.Series)
        mean_series.index = pd.to_datetime(mean_series.index)
        return mean_series.sort_index()

class Analysis:

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir: Path = Path(root_dir)
        self.skt_reader = Era5TemperatureReader(root_dir, "skt")
        self.t2m_reader = Era5TemperatureReader(root_dir, "t2m")

    def plot_daily_timeseries(self) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(
            self.skt_reader.daily_mean.index, self.skt_reader.daily_mean.values,
            color="firebrick", linewidth=0.8
        )
        axes[0].set_title("Daily Skin Temperature")
        axes[0].set_ylabel("Temperature")

        axes[1].plot(
            self.t2m_reader.daily_mean.index, self.t2m_reader.daily_mean.values,
            color="steelblue", linewidth=0.8
        )
        axes[1].set_title("Daily 2m Temperature")
        axes[1].set_ylabel("Temperature")
        axes[1].set_xlabel("Date")
        fig.tight_layout()
        fig.savefig(fname="line_plot.png")
