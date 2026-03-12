from typing import cast, Literal
from functools import cached_property
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd


DATA_ROOT = "/scratch/zgp2ps/era5/raw/singlelevel/"

class Era5TemperatureReader:
    """Load quarterly ERA5 temperature files and concatenate over time."""

    QUARTER_RE: re.Pattern = re.compile(r"^\d{4}q[1-4]$")

    def __init__(
        self, root_dir: str | Path, var_name: Literal["skt", "t2m"], from_year: int, to_year: int
    ) -> None:
        self.root_dir = Path(root_dir)
        self.var_name = var_name
        self.from_year: int = from_year
        self.to_year: int = to_year
        self.file_re: str = "*skin_temperature*" if var_name == "skt" else "*2m_temperature*"

    @cached_property
    def filepaths(self) -> list[Path]:
        quarter_dirs: list[Path] = sorted(
            p for p in self.root_dir.iterdir()
            if p.is_dir()
                and self.QUARTER_RE.match(p.name)
                and self.from_year <= int(p.name[:4]) <= self.to_year
        )
        filepaths: list[Path] = []
        for quarter_dir in quarter_dirs:
            matches: list[Path] = list(quarter_dir.glob(self.file_re))
            assert len(matches) == 1, (
                f"No file found with pattern: {self.file_re}. "
                f"List of available files: {sorted([f.name for f in quarter_dir.glob('*')])}"
            )
            filepaths.append(matches[0])

        return filepaths

    @cached_property
    def dataset(self) -> xr.Dataset:
        ds: xr.Dataset = xr.open_mfdataset(
            self.filepaths,
            engine="h5netcdf",
            combine="nested",
            concat_dim="valid_time",
            parallel=True,
            chunks="auto",
        ).sortby("valid_time")
        ds = ds[[self.var_name]]
        ds = self._select_north_america(ds)
        ds = self._drop_feb29(ds)
        ds = ds.chunk({"valid_time": -1, "latitude": 100, "longitude": 100})
        return ds

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

    @staticmethod
    def _drop_feb29(data: xr.Dataset) -> xr.Dataset:
        time: xr.DataArray = data["valid_time"]
        keep: xr.DataArray = ~((time.dt.month == 2) & (time.dt.day == 29))
        return data.isel(valid_time=keep)

    @staticmethod
    def _select_north_america(data: xr.Dataset) -> xr.Dataset:
        return data.sel(latitude=slice(75, 15), longitude=slice(190, 310))

class DailyMean:

    def __init__(self, skt_reader: Era5TemperatureReader, t2m_reader: Era5TemperatureReader) -> None:
        self.skt_reader = skt_reader
        self.t2m_reader = t2m_reader

    def plot(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            self.skt_reader.daily_mean.index.to_numpy(),
            self.skt_reader.daily_mean.to_numpy(),
            color="firebrick",
            linewidth=0.8,
            label="Skin Temperature",
        )
        ax.plot(
            self.t2m_reader.daily_mean.index.to_numpy(),
            self.t2m_reader.daily_mean.to_numpy(),
            color="steelblue",
            linewidth=0.8,
            label="2m Temperature",
        )
        ax.set_title("Daily Mean Temperature")
        ax.set_ylabel("Temperature")
        ax.set_xlabel("Date")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("daily_mean_lineplot.png", dpi=200)


class HeatwaveAnalysis:

    def __init__(
        self,
        var_name: Literal["skt", "t2m"],
        from_year: int,
        to_year: int,
        percentile: float = 0.95,
        min_days: int = 3,
        min_area_fraction: float = 0.1,
    ) -> None:
        self.var_name: Literal["skt", "t2m"] = var_name
        self.from_year: int = from_year
        self.to_year: int = to_year
        self.percentile: float = percentile
        self.min_days: int = min_days
        self.min_area_fraction: float = min_area_fraction
        self.reader = Era5TemperatureReader(
            root_dir=DATA_ROOT, var_name=var_name, from_year=from_year, to_year=to_year
        )

    @staticmethod
    def _month_day_index(time: pd.DatetimeIndex) -> pd.Index:
        return pd.Index(time.strftime("%m-%d"), name="month_day")

    @staticmethod
    def _month_day_key(data: xr.DataArray) -> xr.DataArray:
        month: xr.DataArray = data["valid_time"].dt.month.astype(str).str.zfill(2)
        day: xr.DataArray = data["valid_time"].dt.day.astype(str).str.zfill(2)
        return month + "-" + day

    @cached_property
    def threshold(self) -> xr.DataArray:
        month_day: xr.DataArray = self._month_day_key(self.reader.dataarray)
        return self.reader.dataarray.groupby(month_day).quantile(self.percentile, dim="valid_time")

    @cached_property
    def extreme_mask(self) -> xr.DataArray:
        month_day: xr.DataArray = self._month_day_key(self.reader.dataarray)
        return self.reader.dataarray.groupby(month_day) > self.threshold

    @cached_property
    def event_table(self) -> pd.DataFrame:
        daily_extreme_fraction: pd.Series = cast(
            pd.Series,
            self.extreme_mask.mean(dim=("latitude", "longitude")).to_pandas(),
        )
        daily_extreme_fraction.index = pd.to_datetime(daily_extreme_fraction.index)
        daily_extreme_fraction = daily_extreme_fraction.sort_index()
        regional_mask: pd.Series = daily_extreme_fraction >= self.min_area_fraction

        regional_mean: pd.Series = cast(
            pd.Series,
            self.reader.dataarray.mean(dim=("latitude", "longitude")).to_pandas(),
        )
        regional_mean.index = pd.to_datetime(regional_mean.index)
        regional_mean = regional_mean.sort_index()

        threshold_mean: pd.Series = cast(
            pd.Series,
            self.threshold.mean(dim=("latitude", "longitude")).to_pandas(),
        )
        threshold_mean.index = pd.Index(threshold_mean.index.astype(str), name="month_day")
        month_day_index: pd.Index = self._month_day_index(pd.DatetimeIndex(regional_mean.index))
        daily_threshold: pd.Series = pd.Series(
            threshold_mean.loc[month_day_index].to_numpy(),
            index=regional_mean.index,
            name="threshold",
        )
        excess: pd.Series = regional_mean - daily_threshold

        events: list[dict[str, object]] = []
        start: pd.Timestamp | None = None
        active_dates: list[pd.Timestamp] = []

        for timestamp, is_event_day in regional_mask.items():
            if is_event_day:
                if start is None:
                    start = timestamp
                    active_dates = []
                active_dates.append(timestamp)
                continue

            values: pd.Series
            anomalies: pd.Series
            if start is not None and len(active_dates) >= self.min_days:
                values = regional_mean.loc[active_dates]
                anomalies = excess.loc[active_dates]
                events.append(
                    {
                        "start": start,
                        "end": active_dates[-1],
                        "duration": len(active_dates),
                        "mean_temp": float(values.mean()),
                        "peak_temp": float(values.max()),
                        "mean_excess": float(anomalies.mean()),
                        "peak_excess": float(anomalies.max()),
                    }
                )

            start = None
            active_dates = []

        if start is not None and len(active_dates) >= self.min_days:
            values = regional_mean.loc[active_dates]
            anomalies = excess.loc[active_dates]
            events.append(
                {
                    "start": start,
                    "end": active_dates[-1],
                    "duration": len(active_dates),
                    "mean_temp": float(values.mean()),
                    "peak_temp": float(values.max()),
                    "mean_excess": float(anomalies.mean()),
                    "peak_excess": float(anomalies.max()),
                }
            )

        return pd.DataFrame(events)

    def summary(self) -> dict[str, pd.Series | pd.DataFrame | xr.DataArray]:
        daily_extreme_fraction: pd.Series = cast(
            pd.Series,
            self.extreme_mask.mean(dim=("latitude", "longitude")).to_pandas(),
        )
        daily_extreme_fraction.index = pd.to_datetime(daily_extreme_fraction.index)
        daily_extreme_fraction = daily_extreme_fraction.sort_index()

        frequency_by_year: pd.Series
        first_event_by_year: pd.Series
        if self.event_table.empty:
            frequency_by_year = pd.Series(dtype=int)
            first_event_by_year = pd.Series(dtype="datetime64[ns]")
        else:
            event_years: pd.Series = pd.to_datetime(self.event_table["start"]).dt.year
            frequency_by_year = event_years.value_counts().sort_index()
            first_event_by_year = cast(
                pd.Series, self.event_table.assign(year=event_years).groupby("year")["start"].min()
            )

        spatial_extreme_frequency: xr.DataArray = self.extreme_mask.mean(dim="valid_time")
        return {
            "daily_extreme_fraction": daily_extreme_fraction,
            "event_table": self.event_table,
            "frequency_by_year": frequency_by_year,
            "first_event_by_year": first_event_by_year,
            "spatial_extreme_frequency": spatial_extreme_frequency,
        }

    def plot_spatial_extreme_frequency(self) -> None:
        spatial_extreme_frequency: xr.DataArray = self.extreme_mask.mean(dim="valid_time")
        values: np.ndarray = spatial_extreme_frequency.to_numpy()

        lon_min: float = float(spatial_extreme_frequency["longitude"].min().item())
        lon_max: float = float(spatial_extreme_frequency["longitude"].max().item())
        lat_min: float = float(spatial_extreme_frequency["latitude"].min().item())
        lat_max: float = float(spatial_extreme_frequency["latitude"].max().item())

        fig, ax = plt.subplots(figsize=(14, 6))
        image = ax.imshow(
            values,
            cmap="hot",
            origin="upper",
            aspect="auto",
            extent=[lon_min, lon_max, lat_min, lat_max],
        )

        world: gpd.GeoDataFrame = gpd.read_file(
            gpd.datasets.get_path("naturalearth_lowres")
        )
        world.boundary.plot(ax=ax, color="black", linewidth=0.4)

        ax.set_title(f"{self.var_name.upper()} Spatial Extreme Frequency", fontsize=18)
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Extreme Frequency", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        fig.tight_layout()
        fig.savefig(f"{self.var_name}_spatial_extreme_frequency.png", dpi=200)



if __name__ == "__main__":

    root: str = "/scratch/zgp2ps/era5/raw/singlelevel/"
    from_year: int = 2023
    to_year: int = 2025
    skt_reader = Era5TemperatureReader(root_dir=root, var_name="skt", from_year=from_year, to_year=to_year)
    t2m_reader = Era5TemperatureReader(root_dir=root, var_name="t2m", from_year=from_year, to_year=to_year)
    # daily_mean = DailyMean(skt_reader=skt_reader, t2m_reader=t2m_reader)
    # daily_mean.plot()
    heatwave = HeatwaveAnalysis(var_name="skt", from_year=from_year, to_year=to_year)
    # heatwave.plot_spatial_extreme_frequency()
