from typing import cast, Literal
from functools import cached_property
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd


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
        datasets: list[xr.Dataset] = [self._drop_feb29(xr.open_dataset(path)) for path in self.filepaths]
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

    @staticmethod
    def _drop_feb29(data: xr.Dataset) -> xr.Dataset:
        time: xr.DataArray = data["valid_time"]
        keep: xr.DataArray = ~((time.dt.month == 2) & (time.dt.day == 29))
        return data.isel(valid_time=keep)


class DailyMean:

    def __init__(self, skt_reader: Era5TemperatureReader, t2m_reader: Era5TemperatureReader) -> None:
        self.skt_reader = skt_reader
        self.t2m_reader = t2m_reader

    def plot(self) -> None:
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

    def plot_spatial_extreme_frequency_globe(self, stride: int = 8) -> None:
        spatial_extreme_frequency: xr.DataArray = self.extreme_mask.mean(dim="valid_time")
        spatial_extreme_frequency = spatial_extreme_frequency.isel(
            latitude=slice(None, None, stride),
            longitude=slice(None, None, stride),
        )

        lon_deg: np.ndarray = spatial_extreme_frequency["longitude"].to_numpy()
        lat_deg: np.ndarray = spatial_extreme_frequency["latitude"].to_numpy()
        lon2d_deg: np.ndarray
        lat2d_deg: np.ndarray
        lon2d_deg, lat2d_deg = np.meshgrid(lon_deg, lat_deg)

        lon_rad: np.ndarray = np.deg2rad(lon2d_deg)
        lat_rad: np.ndarray = np.deg2rad(lat2d_deg)

        radius: float = 1.0
        x: np.ndarray = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y: np.ndarray = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z: np.ndarray = radius * np.sin(lat_rad)

        values: np.ndarray = spatial_extreme_frequency.to_numpy()
        norm = plt.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values))) # pyright: ignore
        facecolors = plt.cm.hot(norm(values)) # pyright: ignore

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        mappable = plt.cm.ScalarMappable(cmap="hot", norm=norm)
        mappable.set_array(values)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label="Extreme Frequency")

        ax.set_title(f"{self.var_name.upper()} Spatial Extreme Frequency on Globe")
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(f"{self.var_name}_extreme_frequency_globe.png", dpi=200)



if __name__ == "__main__":

    root: str = "/scratch/zgp2ps/era5/raw/singlelevel/"
    from_year: int = 2020
    to_year: int = 2025
    skt_reader = Era5TemperatureReader(root_dir=root, var_name="skt", from_year=from_year, to_year=to_year)
    t2m_reader = Era5TemperatureReader(root_dir=root, var_name="t2m", from_year=from_year, to_year=to_year)
    daily_mean = DailyMean(skt_reader=skt_reader, t2m_reader=t2m_reader)
    daily_mean.plot()
    heatwave = HeatwaveAnalysis(var_name="skt", from_year=from_year, to_year=to_year)
    heatwave.plot_spatial_extreme_frequency_globe()
