from typing import cast, Literal, Any
from functools import cached_property, cache
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import gaussian_kde
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmasher as cmr
from cmap import Colormap

DATA_ROOT = "/scratch/zgp2ps/era5/raw/singlelevel/"
LAND_MASK_PATH = "/scratch/zgp2ps/era5/raw/landmask/landmask.nc"


class Era5TemperatureReader:

    QUARTER_RE: re.Pattern = re.compile(r"^\d{4}q[1-4]$")

    def __init__(
        self, root_dir: str | Path,
        var_name: Literal["skt", "t2m"],
        from_year: int, to_year: int
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
            ds = cast(xr.Dataset, self._select_conus(ds))
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
        mask = cast(xr.DataArray, self._select_conus(mask))
        return mask >= 0.5

    @cached_property
    def month_day_key(self) -> xr.DataArray:
        month: xr.DataArray = self.dataarray["valid_time"].dt.month
        day: xr.DataArray = self.dataarray["valid_time"].dt.day
        return month * 100 + day

    @cached_property
    def climatology(self) -> xr.DataArray:
        # 2D array
        climatology = self.dataarray.groupby(self.month_day_key).mean(dim="valid_time", skipna=True)
        return climatology

    @cached_property
    def anomalies(self) -> xr.DataArray:
        # 3D array
        anomalies: xr.DataArray = self.dataarray.groupby(self.month_day_key) - self.climatology # broadcast
        return anomalies

    @staticmethod
    def _drop_feb29(data: xr.Dataset) -> xr.Dataset:
        time: xr.DataArray = data["valid_time"]
        keep: xr.DataArray = ~((time.dt.month == 2) & (time.dt.day == 29))
        return data.isel(valid_time=keep)

    @staticmethod
    def _select_conus(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        return data.sel(
            latitude=slice(49.5, 24.0),
            longitude=slice(235.0, 293.5),
        )


class DailyMean:

    def __init__(self, skt_reader: Era5TemperatureReader, t2m_reader: Era5TemperatureReader) -> None:
        self.skt_reader = skt_reader
        self.t2m_reader = t2m_reader

    @cache
    def daily_mean(self, var_name: Literal["skt", "t2m"]) -> pd.Series:
        mean: xr.DataArray
        if var_name == "skt":
            mean = self.skt_reader.dataarray.mean(dim=("latitude", "longitude"))
        else:
            mean = self.t2m_reader.dataarray.mean(dim=("latitude", "longitude"))
        mean_series: pd.Series = cast(pd.Series, mean.to_pandas())
        assert isinstance(mean_series, pd.Series)
        mean_series.index = pd.to_datetime(mean_series.index)
        return mean_series.sort_index()

    def plot(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 6))
        skt_mean: pd.Series = self.daily_mean("skt")
        ax.plot(
            skt_mean.index.to_numpy(),
            skt_mean.to_numpy(),
            color="firebrick", linewidth=0.8,
            label="Skin Temperature",
        )
        t2m_mean: pd.Series = self.daily_mean("t2m")
        ax.plot(
            t2m_mean.index.to_numpy(),
            t2m_mean.to_numpy(),
            color="steelblue",
            linewidth=0.8,
            label="2m Temperature",
        )
        ax.set_title("Daily Mean Temperature", fontsize=16)
        ax.set_ylabel("Temperature", fontsize=14)
        ax.set_xlabel("Date", fontsize=14)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("daily_mean_lineplot.png", dpi=200)


class HeatwaveAnalysis:

    def __init__(
        self,
        var_name: Literal["skt", "t2m"],
        from_year: int,
        to_year: int,
        percentile: float = 0.9,
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
            q=self.percentile, dim="valid_time", skipna=True,
        )
        return threshold.squeeze(drop=True)

    @cached_property
    def extreme_mask(self) -> xr.DataArray:
        extreme_mask: xr.DataArray = self.reader.dataarray > self.threshold
        return extreme_mask.squeeze(drop=True)

    @cached_property
    def yearly_count_map(self) -> xr.DataArray:
        return (
            self.extreme_mask
            .where(self.reader.land_mask)
            .groupby("valid_time.year")
            .sum(dim="valid_time", skipna=True)
            .where(self.reader.land_mask)
            .squeeze(drop=True)
        )

    @cached_property
    def daily_extreme_spatial_fraction(self) -> xr.DataArray:
        extreme_on_land: xr.DataArray = self.extreme_mask.where(self.reader.land_mask)
        extreme_count: xr.DataArray = extreme_on_land.sum(dim=("latitude", "longitude"))
        land_count: xr.DataArray = self.reader.land_mask.sum(dim=("latitude", "longitude"))
        return extreme_count / land_count

    @cached_property
    def frequency_by_year(self) -> pd.Series:
        regional_mask: xr.DataArray = self.daily_extreme_spatial_fraction >= self.min_area_fraction
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
                    # End of heatwave -> record start_index
                    event_starts.append(cast(pd.Timestamp, time[start_index]))
                # Not classified as heatwave -> reset and move on
                start_index = None
                run_length = 0

        # End of data but still in the middle of a heatwave -> record start_index
        if start_index is not None and run_length >= self.min_days:
            event_starts.append(cast(pd.Timestamp, time[start_index]))

        if not event_starts:
            # No heatwaves detected -> return empty pd.Series
            return pd.Series(dtype=int)

        # Heatwaves detected -> return pd.Series of yearly frequency
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

        fig = plt.figure(figsize=(9.5, 6))
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
            vmax=90,
            transform=ccrs.PlateCarree(),
        )
        ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="black")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="black")
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="black")
        ax.set_title(f"Number of Extreme Days - {year}", fontsize=18)
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02, fraction=0.035, aspect=30)
        cbar.set_label("Day Count", fontsize=16)
        fig.tight_layout()
        fig.savefig(f"{self.var_name}_count_map_{year}.png", dpi=400)


class TemperaturePersistenceAnalysis:

    def __init__(self, skt_reader: Era5TemperatureReader, t2m_reader: Era5TemperatureReader) -> None:
        self.skt_reader = skt_reader
        self.t2m_reader = t2m_reader

    @staticmethod
    def autocorrelation(reader: Era5TemperatureReader, lag: int) -> xr.DataArray:
        anomaly_on_land: xr.DataArray = reader.anomalies.where(reader.land_mask)
        lead_arrays = anomaly_on_land.isel(valid_time=slice(None, -lag))
        lag_arrays = anomaly_on_land.isel(valid_time=slice(lag, None))
        lag_arrays = lag_arrays.assign_coords(valid_time=lead_arrays.valid_time)
        assert lead_arrays.shape == lag_arrays.shape
        pixelwise_corr: xr.DataArray = xr.corr(lead_arrays, lag_arrays, dim="valid_time")
        return pixelwise_corr

    @staticmethod
    def _plot_autocorrelation_panel(ax: Any, autocorr: xr.DataArray, title: str) -> Any:
        if float(autocorr.longitude.max()) > 180:
            autocorr = autocorr.assign_coords(
                longitude=((autocorr.longitude + 180) % 360) - 180
            ).sortby("longitude")

        lon: np.ndarray = autocorr["longitude"].to_numpy()
        lat: np.ndarray = autocorr["latitude"].to_numpy()
        values: np.ndarray = autocorr.to_numpy()
        mesh = ax.pcolormesh(
            lon,
            lat,
            values,
            shading="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            transform=ccrs.PlateCarree(),
        )
        ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="black")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="black")
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="black")
        ax.set_title(title, fontsize=22)
        ax.set_xlabel("Longitude", fontsize=18)
        ax.set_ylabel("Latitude", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        return mesh

    def plot_autocorrelation(self, max_lag: int = 30) -> None:
        for lag in range(1, max_lag + 1):
            skt_autocorr = self.autocorrelation(reader=self.skt_reader, lag=lag)
            t2m_autocorr = self.autocorrelation(reader=self.t2m_reader, lag=lag)
            fig, axes = plt.subplots(
                1, 2,
                figsize=(19, 6),
                subplot_kw={
                    "projection": ccrs.AlbersEqualArea(
                        central_longitude=-96,
                        central_latitude=37.5,
                    )
                },
            )
            fig.subplots_adjust(left=0.02, right=0.995, top=0.9, bottom=0.1, wspace=0.01)
            mesh = self._plot_autocorrelation_panel(
                axes[0],
                skt_autocorr,
                f"SKT Anomaly Autocorrelation (lag={lag} days)",
            )
            self._plot_autocorrelation_panel(
                axes[1],
                t2m_autocorr,
                f"T2M Anomaly Autocorrelation (lag={lag} days)",
            )
            cbar = fig.colorbar(
                mesh,
                ax=axes,
                orientation="horizontal",
                shrink=0.8,
                pad=0.035,
                fraction=0.035,
                aspect=40,
            )
            cbar.set_label("Correlation", fontsize=18)
            cbar.ax.tick_params(labelsize=14)
            fig.savefig(f"temperature_anomaly_autocorrelation_comparison_lag_{lag}.png", dpi=300)


class SameTimeHeatwaveAnalysis(HeatwaveAnalysis):

    @cached_property
    def threshold(self) -> xr.DataArray:
        """
        Override default threshold logic. Now, it consider same-time of year only
        """
        threshold: xr.DataArray = self.reader.dataarray.groupby(self.reader.month_day_key).quantile(
            self.percentile,
            dim="valid_time",
            skipna=True,
        )
        return threshold.squeeze(drop=True) # (365 x lat x lon)

    @cached_property
    def extreme_mask(self) -> xr.DataArray:
        extreme_mask: xr.DataArray = self.reader.dataarray.groupby(self.reader.month_day_key) > self.threshold  # broadcast
        return extreme_mask.squeeze(drop=True)

    @staticmethod
    def _week_index_from_dayofyear(dayofyear: pd.Series) -> pd.Series:
        # Map 365 days into exactly 52 week bins.
        return ((dayofyear - 1) * 52 // 365) + 1

    @cached_property
    def weekly_extreme_day_counts(self) -> pd.DataFrame:
        regional_mask: xr.DataArray = self.daily_extreme_spatial_fraction >= self.min_area_fraction # calculated base on modified threshold
        event_series: pd.Series = cast(pd.Series, regional_mask.to_pandas()).astype(int)
        event_series.index = pd.to_datetime(event_series.index)
        heatmap_frame = pd.DataFrame(
            {
                "year": event_series.index.year,    # pyright: ignore
                "week": self._week_index_from_dayofyear(pd.Series(event_series.index.dayofyear, index=event_series.index)), # pyright: ignore
                "extreme_day": event_series.to_numpy(),
            },
            index=event_series.index,
        )
        weekly_counts = (
            heatmap_frame
            .groupby(["year", "week"])["extreme_day"]
            .sum()
            .unstack(fill_value=0)  # pyright: ignore
            .reindex(index=range(self.from_year, self.to_year + 1), fill_value=0)
            .reindex(columns=range(1, 53), fill_value=0)  # pyright: ignore
        )
        return weekly_counts

    def plot_heatwave_distribution(self) -> None:
        weekly_counts: pd.DataFrame = self.weekly_extreme_day_counts
        values: np.ndarray = weekly_counts.to_numpy()
        discrete_cmap = colors.ListedColormap(cmr.sunburst_r(np.linspace(0.1, 0.95, 8)))
        discrete_norm = colors.BoundaryNorm(np.arange(-0.5, 8.5, 1), discrete_cmap.N)
        fig_width = max(12, weekly_counts.shape[1] * 0.25)
        fig_height = max(6, weekly_counts.shape[0] * 0.25)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(
            values,
            aspect="equal",
            cmap=discrete_cmap,
            norm=discrete_norm,
            interpolation="nearest",
        )
        ax.set_title(f"{self.var_name.upper()} Heatwave Distribution by Week", fontsize=18)
        ax.set_xlabel("Week of Year", fontsize=14)
        ax.set_ylabel("Year", fontsize=14)
        ax.set_xticks(np.arange(52))
        ax.set_xticklabels(np.arange(1, 53))
        ax.set_yticks(np.arange(len(weekly_counts.index)))
        ax.set_yticklabels(weekly_counts.index.astype(str))
        ax.set_xticks(np.arange(-0.5, weekly_counts.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, weekly_counts.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="x", labelsize=8)
        cbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.035, aspect=30, ticks=np.arange(8))
        cbar.set_label("Extreme Days per Week", fontsize=14)
        fig.tight_layout()
        fig.savefig(f"{self.var_name}_heatwave_distribution_by_week.png", dpi=400)


class SkinAirTemperatureDifferenceAnalysis:

    def __init__(
        self,
        skt_reader: Era5TemperatureReader,
        t2m_reader: Era5TemperatureReader,
        heatwave_analysis: SameTimeHeatwaveAnalysis,
    ) -> None:
        self.skt_reader = skt_reader
        self.t2m_reader = t2m_reader
        self.heatwave_analysis = heatwave_analysis

    @cached_property
    def skin_air_difference(self) -> xr.DataArray:
        return self.skt_reader.dataarray - self.t2m_reader.dataarray

    @cached_property
    def regional_heatwave_mask(self) -> xr.DataArray:
        return self.heatwave_analysis.daily_extreme_spatial_fraction >= self.heatwave_analysis.min_area_fraction

    @cached_property
    def daily_mean_difference(self) -> pd.Series:
        land_mask: xr.DataArray = self.skt_reader.land_mask
        mean_difference = self.skin_air_difference.where(land_mask).mean(dim=("latitude", "longitude"), skipna=True)
        mean_series: pd.Series = cast(pd.Series, mean_difference.to_pandas())
        mean_series.index = pd.to_datetime(mean_series.index)
        return mean_series.sort_index()

    @cached_property
    def heatwave_normal_difference(self) -> tuple[pd.Series, pd.Series]:
        heatwave_mask = cast(pd.Series, self.regional_heatwave_mask.to_pandas()).astype(bool)
        heatwave_mask.index = pd.to_datetime(heatwave_mask.index)
        aligned_difference = self.daily_mean_difference.reindex(heatwave_mask.index)
        heatwave_difference = aligned_difference.loc[heatwave_mask]
        normal_difference = aligned_difference.loc[~heatwave_mask]
        return heatwave_difference, normal_difference

    def plot_distribution(self) -> None:
        heatwave_difference, normal_difference = self.heatwave_normal_difference
        fig, ax = plt.subplots(figsize=(10, 6))
        x_min = min(float(normal_difference.min()), float(heatwave_difference.min()))
        x_max = max(float(normal_difference.max()), float(heatwave_difference.max()))
        x_grid = np.linspace(x_min, x_max, 400)
        bins = np.linspace(x_min, x_max, 30)
        normal_kde = gaussian_kde(normal_difference.to_numpy())
        heatwave_kde = gaussian_kde(heatwave_difference.to_numpy())
        ax.hist(
            normal_difference.to_numpy(),
            bins=bins,  # pyright: ignore
            density=True,
            color="steelblue",
            alpha=0.25,
            label="Normal Days",
        )
        ax.hist(
            heatwave_difference.to_numpy(),
            bins=bins,  # pyright: ignore
            density=True,
            color="firebrick",
            alpha=0.25,
            label="Heatwave Days",
        )
        ax.plot(
            x_grid,
            normal_kde(x_grid),
            color="steelblue",
            linewidth=2,
        )
        ax.plot(
            x_grid,
            heatwave_kde(x_grid),
            color="firebrick",
            linewidth=2,
        )
        ax.fill_between(x_grid, normal_kde(x_grid), color="steelblue", alpha=0.2)
        ax.fill_between(x_grid, heatwave_kde(x_grid), color="firebrick", alpha=0.2)
        ax.axvline(float(normal_difference.mean()), color="steelblue", linestyle="--", linewidth=1.5)
        ax.axvline(float(heatwave_difference.mean()), color="firebrick", linestyle="--", linewidth=1.5)
        ax.set_title("Distribution of SKT - T2M", fontsize=16)
        ax.set_xlabel("SKT - T2M", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("skt_minus_t2m_distribution_heatwave_vs_normal.png", dpi=300)


if __name__ == "__main__":

    root: str = "/scratch/zgp2ps/era5/raw/singlelevel/"

    # skt_reader = Era5TemperatureReader(root_dir=root, var_name="skt", from_year=2025, to_year=2025)
    # t2m_reader = Era5TemperatureReader(root_dir=root, var_name="t2m", from_year=2025, to_year=2025)
    # daily_mean = DailyMean(skt_reader=skt_reader, t2m_reader=t2m_reader)
    # daily_mean.plot()

    from_year: int = 2000
    to_year: int = 2025
    skt_reader = Era5TemperatureReader(root_dir=root, var_name="skt", from_year=from_year, to_year=to_year)
    t2m_reader = Era5TemperatureReader(root_dir=root, var_name="t2m", from_year=from_year, to_year=to_year)

    # persistence = TemperaturePersistenceAnalysis(skt_reader=skt_reader, t2m_reader=t2m_reader)
    # persistence.plot_autocorrelation(max_lag=30)

    skt_heatwave = SameTimeHeatwaveAnalysis(var_name="skt", from_year=from_year, to_year=to_year)
    t2m_heatwave = SameTimeHeatwaveAnalysis(var_name="t2m", from_year=from_year, to_year=to_year)

    # skt_heatwave.plot_frequency_by_year()
    # t2m_heatwave.plot_frequency_by_year()

    skt_t2m_difference = SkinAirTemperatureDifferenceAnalysis(
        skt_reader=skt_reader,
        t2m_reader=t2m_reader,
        heatwave_analysis=skt_heatwave,
    )
    skt_t2m_difference.plot_distribution()

    # for year in range(from_year, to_year + 1):
    #     skt_heatwave.plot_count_map(year)
    #     t2m_heatwave.plot_count_map(year)

    # skt_heatwave.plot_heatwave_distribution()
    # t2m_heatwave.plot_heatwave_distribution()
