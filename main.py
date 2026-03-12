from typing import cast, Literal, Any
from functools import cached_property
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmasher as cmr

DATA_ROOT = "/scratch/zgp2ps/era5/raw/singlelevel/"
LAND_MASK_PATH = "/scratch/zgp2ps/era5/raw/landmask/landmask.nc"


class Era5TemperatureReader:

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
        datasets: list[xr.Dataset] = []
        for path in self.filepaths:
            ds: xr.Dataset = xr.open_dataset(path, engine="h5netcdf")
            ds = ds[[self.var_name]]
            ds = self._drop_feb29(ds)
            ds = self._select_conus(ds)
            datasets.append(ds)
        combined: xr.Dataset = xr.concat(datasets, dim="valid_time").sortby("valid_time")
        return combined

    @cached_property
    def dataarray(self) -> xr.DataArray:
        return self.dataset[self.var_name]

    @cached_property
    def land_mask(self) -> xr.DataArray:
        ds: xr.Dataset = xr.open_dataset(LAND_MASK_PATH, engine="h5netcdf")
        mask: xr.DataArray = ds["lsm"].squeeze(drop=True)
        mask = self._select_conus(mask)
        return mask >= 0.5

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
    def _select_conus(data: xr.Dataset) -> xr.Dataset:
        return data.sel(
            latitude=slice(49.5, 24.0),
            longitude=slice(235.0, 293.5),
        )

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

    @cached_property
    def threshold(self) -> xr.DataArray:
        threshold: xr.DataArray = self.reader.dataarray.quantile(
            self.percentile,
            dim="valid_time",
            skipna=True,
        )
        return threshold.squeeze(drop=True)

    @cached_property
    def extreme_mask(self) -> xr.DataArray:
        extreme_mask: xr.DataArray = self.reader.dataarray > self.threshold
        return extreme_mask.squeeze(drop=True)

    @cached_property
    def yearly_count_map(self) -> xr.DataArray:
        land_mask: xr.DataArray = self.reader.land_mask
        yearly_count_map: xr.DataArray = (
            self.extreme_mask
            .where(land_mask)
            .groupby("valid_time.year")
            .sum(dim="valid_time", skipna=True)
            .where(land_mask)
            .squeeze(drop=True)
        )
        return yearly_count_map

    @cached_property
    def daily_extreme_fraction(self) -> xr.DataArray:
        land_mask: xr.DataArray = self.reader.land_mask
        extreme_on_land: xr.DataArray = self.extreme_mask.where(land_mask)
        land_count: xr.DataArray = land_mask.sum(dim=("latitude", "longitude"))
        extreme_count: xr.DataArray = extreme_on_land.sum(dim=("latitude", "longitude"))
        return extreme_count / land_count

    @cached_property
    def frequency_by_year(self) -> pd.Series:
        regional_mask: xr.DataArray = self.daily_extreme_fraction >= self.min_area_fraction
        mask: np.ndarray = regional_mask.to_numpy()
        time: pd.DatetimeIndex = pd.to_datetime(regional_mask["valid_time"].to_numpy())
        event_starts: list[pd.Timestamp] = []
        run_length: int = 0
        start_index: int | None = None
        for index, is_event_day in enumerate(mask):
            if bool(is_event_day):
                if run_length == 0:
                    start_index = index
                run_length += 1
            else:
                if start_index is not None and run_length >= self.min_days:
                    event_starts.append(cast(pd.Timestamp, time[start_index]))
                start_index = None
                run_length = 0

        if start_index is not None and run_length >= self.min_days:
            event_starts.append(cast(pd.Timestamp, time[start_index]))

        if not event_starts:
            return pd.Series(dtype=int)

        years: pd.Index = pd.Index([timestamp.year for timestamp in event_starts], name="year")
        frequency: pd.Series = years.value_counts().sort_index()
        return frequency

    def plot_frequency_by_year(self) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(self.frequency_by_year.index.to_numpy(), self.frequency_by_year.to_numpy(), color="steelblue", width=0.8)
        ax.set_title(f"{self.var_name.upper()} Heatwave Frequency by Year", fontsize=18)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Number of Heatwaves", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{self.var_name}_heatwave_frequency_by_year.png", dpi=200)

    def plot_count_map(self, year: int) -> None:
        count_map: xr.DataArray = self.yearly_count_map.sel(year=year)

        if float(count_map.longitude.max()) > 180:
            count_map = count_map.assign_coords(
                longitude=((count_map.longitude + 180) % 360) - 180
            ).sortby("longitude")

        lon: np.ndarray = count_map["longitude"].to_numpy()
        lat: np.ndarray = count_map["latitude"].to_numpy()
        values: np.ndarray = count_map.to_numpy()

        fig = plt.figure(figsize=(10, 6))
        ax: Any = fig.add_subplot(
            111,
            projection=ccrs.AlbersEqualArea(
                central_longitude=-96,
                central_latitude=37.5,
            ),
        )
        mesh = ax.pcolormesh(
            lon,
            lat,
            values,
            shading="auto",
            cmap=cmr.sunburst_r,
            vmin=0,
            vmax=50,
            transform=ccrs.PlateCarree(),
        )
        ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="black")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="black")
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="black")
        ax.set_title(f"Count of Extreme Days per Pixel ({year})", fontsize=18)
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)

        cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.02, fraction=0.035, aspect=30)
        cbar.set_label("Extreme Day Count", fontsize=14)

        fig.tight_layout()
        fig.savefig(f"{self.var_name}_count_map_{year}.png", dpi=400)

class SameTimeHeatwaveAnalysis(HeatwaveAnalysis):

    @cached_property
    def month_day_key(self) -> xr.DataArray:
        month: xr.DataArray = self.reader.dataarray["valid_time"].dt.month
        day: xr.DataArray = self.reader.dataarray["valid_time"].dt.day
        return month * 100 + day

    @cached_property
    def threshold(self) -> xr.DataArray:
        threshold: xr.DataArray = self.reader.dataarray.groupby(self.month_day_key).quantile(
            self.percentile,
            dim="valid_time",
            skipna=True,
        )
        return threshold.squeeze(drop=True)

    @cached_property
    def extreme_mask(self) -> xr.DataArray:
        extreme_mask: xr.DataArray = self.reader.dataarray.groupby(self.month_day_key) > self.threshold
        return extreme_mask.squeeze(drop=True)

    @staticmethod
    def _week_index_from_dayofyear(dayofyear: pd.Series) -> pd.Series:
        # Map 365 days into exactly 52 week bins.
        return ((dayofyear - 1) * 52 // 365) + 1

    @staticmethod
    def _month_tick_positions() -> tuple[list[float], list[str]]:
        month_starts = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
        start_weeks = [int(((timestamp.dayofyear - 1) * 52 // 365) + 1) for timestamp in month_starts]
        next_start_weeks = start_weeks[1:] + [53]
        tick_positions = [
            (start_week - 1 + next_week - 2) / 2 for start_week, next_week in zip(start_weeks, next_start_weeks)
        ]
        tick_labels = [timestamp.strftime("%b") for timestamp in month_starts]
        return tick_positions, tick_labels

    @cached_property
    def weekly_extreme_day_counts(self) -> pd.DataFrame:
        regional_mask: xr.DataArray = self.daily_extreme_fraction >= self.min_area_fraction
        event_series: pd.Series = cast(pd.Series, regional_mask.to_pandas()).astype(int)
        event_series.index = pd.to_datetime(event_series.index)
        heatmap_frame = pd.DataFrame(
            {
                "year": event_series.index.year,
                "week": self._week_index_from_dayofyear(pd.Series(event_series.index.dayofyear, index=event_series.index)),
                "extreme_day": event_series.to_numpy(),
            },
            index=event_series.index,
        )
        weekly_counts = (
            heatmap_frame
            .groupby(["year", "week"])["extreme_day"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=range(self.from_year, self.to_year + 1), fill_value=0)
            .reindex(columns=range(1, 53), fill_value=0)
        )
        return weekly_counts

    def plot_heatwave_distribution(self) -> None:
        weekly_counts: pd.DataFrame = self.weekly_extreme_day_counts
        values: np.ndarray = weekly_counts.to_numpy()

        fig_width = max(12, weekly_counts.shape[1] * 0.25)
        fig_height = max(6, weekly_counts.shape[0] * 0.25)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(
            values,
            aspect="equal",
            cmap=cmr.sunburst_r,
            vmin=0,
            vmax=7,
            interpolation="nearest",
        )

        ax.set_title(f"{self.var_name.upper()} Heatwave Distribution by Week", fontsize=18)
        ax.set_xlabel("Week of Year", fontsize=14)
        ax.set_ylabel("Year", fontsize=14)
        ax.set_xticks(np.arange(0, 52, 4))
        ax.set_xticklabels(np.arange(1, 53, 4))
        ax.set_yticks(np.arange(len(weekly_counts.index)))
        ax.set_yticklabels(weekly_counts.index.astype(str))
        ax.set_xticks(np.arange(-0.5, weekly_counts.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, weekly_counts.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        top_ax = ax.twiny()
        top_ax.set_xlim(ax.get_xlim())
        month_positions, month_labels = self._month_tick_positions()
        top_ax.set_xticks(month_positions)
        top_ax.set_xticklabels(month_labels)
        top_ax.set_xlabel("Month", fontsize=14)

        cbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.035, aspect=30)
        cbar.set_label("Extreme Days per Week", fontsize=14)

        fig.tight_layout()
        fig.savefig(f"{self.var_name}_heatwave_distribution_by_week.png", dpi=400)


if __name__ == "__main__":

    root: str = "/scratch/zgp2ps/era5/raw/singlelevel/"
    from_year: int = 2000
    to_year: int = 2025
    skt_reader = Era5TemperatureReader(root_dir=root, var_name="skt", from_year=from_year, to_year=to_year)
    t2m_reader = Era5TemperatureReader(root_dir=root, var_name="t2m", from_year=from_year, to_year=to_year)
    # daily_mean = DailyMean(skt_reader=skt_reader, t2m_reader=t2m_reader)
    # daily_mean.plot()
    heatwave = SameTimeHeatwaveAnalysis(var_name="skt", from_year=from_year, to_year=to_year)
    heatwave.reader.dataset
    heatwave.threshold
    # heatwave.frequency_by_year
    # heatwave.plot_frequency_by_year()
    # for year in range(from_year, to_year + 1):
    #     heatwave.plot_count_map(year)
    #
    heatwave.plot_heatwave_distribution()
