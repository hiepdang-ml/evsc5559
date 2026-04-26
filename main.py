from typing import cast, Literal, Any
from functools import cached_property, cache
from pathlib import Path
import re
import math

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import gaussian_kde, linregress
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmasher as cmr

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

    BOOTSTRAP_SEED: int = 42
    MONTH_TO_SEASON: dict[int, str] = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }
    REGION_LATITUDE_SPLIT: float = (49.5 + 24.0) / 2
    REGION_LONGITUDE_SPLIT: float = (235.0 + 293.5) / 2
    REGION_ORDER: tuple[str, ...] = ("South-west", "South-east", "North-west", "North-east")

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

    @cached_property
    def daily_lag_analysis_frame(self) -> pd.DataFrame:
        heatwave_mask = cast(pd.Series, self.regional_heatwave_mask.to_pandas()).astype(int)
        heatwave_mask.index = pd.to_datetime(heatwave_mask.index)
        aligned_difference = self.daily_mean_difference.reindex(heatwave_mask.index)
        frame = pd.DataFrame(
            {
                "skt_minus_t2m": aligned_difference,
                "heatwave": heatwave_mask,
            },
            index=heatwave_mask.index,
        ).dropna()
        return frame.sort_index()

    @cached_property
    def region_masks(self) -> dict[str, xr.DataArray]:
        land_mask = self.skt_reader.land_mask
        latitude = land_mask["latitude"]
        longitude = land_mask["longitude"]
        south_mask = latitude < self.REGION_LATITUDE_SPLIT
        north_mask = latitude >= self.REGION_LATITUDE_SPLIT
        west_mask = longitude < self.REGION_LONGITUDE_SPLIT
        east_mask = longitude >= self.REGION_LONGITUDE_SPLIT
        return {
            "South-west": land_mask & south_mask & west_mask,
            "South-east": land_mask & south_mask & east_mask,
            "North-west": land_mask & north_mask & west_mask,
            "North-east": land_mask & north_mask & east_mask,
        }

    @cached_property
    def seasonal_heatwave_normal_difference(self) -> dict[str, tuple[pd.Series, pd.Series]]:
        heatwave_difference, normal_difference = self.heatwave_normal_difference
        seasonal_difference: dict[str, tuple[pd.Series, pd.Series]] = {}
        for season in ("Winter", "Spring", "Summer", "Fall"):
            season_months = [
                month for month, mapped_season in self.MONTH_TO_SEASON.items()
                if mapped_season == season
            ]
            seasonal_heatwave = heatwave_difference.loc[heatwave_difference.index.month.isin(season_months)] # pyright: ignore
            seasonal_normal = normal_difference.loc[normal_difference.index.month.isin(season_months)]       # pyright: ignore
            seasonal_difference[season] = (seasonal_heatwave, seasonal_normal)
        return seasonal_difference

    @cached_property
    def regional_heatwave_normal_difference(self) -> dict[str, tuple[pd.Series, pd.Series]]:
        regional_difference: dict[str, tuple[pd.Series, pd.Series]] = {}
        for region in self.REGION_ORDER:
            region_mask = self.region_masks[region]
            regional_mean_difference = self.skin_air_difference.where(region_mask).mean(
                dim=("latitude", "longitude"),
                skipna=True,
            )
            difference_series: pd.Series = cast(pd.Series, regional_mean_difference.to_pandas())
            difference_series.index = pd.to_datetime(difference_series.index)
            difference_series = difference_series.sort_index()

            regional_extreme = self.heatwave_analysis.extreme_mask.where(region_mask)
            extreme_count = regional_extreme.sum(dim=("latitude", "longitude"), skipna=True)
            land_count = region_mask.sum(dim=("latitude", "longitude"))
            regional_fraction = extreme_count / land_count
            regional_heatwave_mask = cast(pd.Series, regional_fraction.to_pandas()).fillna(0).astype(float)
            regional_heatwave_mask.index = pd.to_datetime(regional_heatwave_mask.index)
            is_heatwave = regional_heatwave_mask >= self.heatwave_analysis.min_area_fraction

            aligned_difference = difference_series.reindex(regional_heatwave_mask.index)
            heatwave_difference = aligned_difference.loc[is_heatwave]
            normal_difference = aligned_difference.loc[~is_heatwave]
            regional_difference[region] = (heatwave_difference, normal_difference)
        return regional_difference

    @staticmethod
    def _decade_label(year: int) -> str:
        decade_start = (year // 10) * 10
        return f"{decade_start}s"

    @cached_property
    def decadal_seasonal_heatwave_normal_difference(self) -> dict[str, dict[str, tuple[pd.Series, pd.Series]]]:
        heatwave_difference, normal_difference = self.heatwave_normal_difference
        decade_labels = sorted(
            {self._decade_label(year) for year in self.daily_mean_difference.index.year},  # pyright: ignore
        )
        decadal_difference: dict[str, dict[str, tuple[pd.Series, pd.Series]]] = {}
        for decade in decade_labels:
            decade_start = int(decade[:-1])
            decade_end = decade_start + 9
            decadal_heatwave = heatwave_difference.loc[
                (heatwave_difference.index.year >= decade_start) & (heatwave_difference.index.year <= decade_end)  # pyright: ignore
            ]
            decadal_normal = normal_difference.loc[
                (normal_difference.index.year >= decade_start) & (normal_difference.index.year <= decade_end)  # pyright: ignore
            ]
            seasonal_difference: dict[str, tuple[pd.Series, pd.Series]] = {}
            for season in ("Winter", "Spring", "Summer", "Fall"):
                season_months = [
                    month for month, mapped_season in self.MONTH_TO_SEASON.items()
                    if mapped_season == season
                ]
                seasonal_heatwave = decadal_heatwave.loc[decadal_heatwave.index.month.isin(season_months)]  # pyright: ignore
                seasonal_normal = decadal_normal.loc[decadal_normal.index.month.isin(season_months)]  # pyright: ignore
                seasonal_difference[season] = (seasonal_heatwave, seasonal_normal)
            decadal_difference[decade] = seasonal_difference
        return decadal_difference

    def lagged_relationships(self, max_lag: int = 7) -> pd.DataFrame:
        frame = self.daily_lag_analysis_frame
        results: list[dict[str, float | int]] = []
        for lag in range(-max_lag, max_lag + 1):
            shifted_heatwave = frame["heatwave"].shift(-lag)
            lagged_frame = pd.DataFrame(
                {
                    "skt_minus_t2m": frame["skt_minus_t2m"],
                    "heatwave": shifted_heatwave,
                },
                index=frame.index,
            ).dropna()
            if len(lagged_frame) < 3:
                continue

            predictor = lagged_frame["skt_minus_t2m"].to_numpy(dtype=float)
            response = lagged_frame["heatwave"].to_numpy(dtype=float)
            correlation = float(np.corrcoef(predictor, response)[0, 1])
            slope, intercept = np.polyfit(predictor, response, deg=1)
            response_std = float(response.std(ddof=0))
            if math.isclose(response_std, 0.0):
                mean_heatwave_given_high = float("nan")
                mean_heatwave_given_low = float("nan")
            else:
                high_threshold = float(np.quantile(predictor, 0.75))
                low_threshold = float(np.quantile(predictor, 0.25))
                high_mask = lagged_frame["skt_minus_t2m"] >= high_threshold
                low_mask = lagged_frame["skt_minus_t2m"] <= low_threshold
                mean_heatwave_given_high = float(lagged_frame.loc[high_mask, "heatwave"].mean())
                mean_heatwave_given_low = float(lagged_frame.loc[low_mask, "heatwave"].mean())

            results.append(
                {
                    "lag_days": lag,
                    "correlation": correlation,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": correlation ** 2,
                    "sample_size": len(lagged_frame),
                    "heatwave_rate_high_skt_minus_t2m": mean_heatwave_given_high,
                    "heatwave_rate_low_skt_minus_t2m": mean_heatwave_given_low,
                }
            )

        return pd.DataFrame(results).set_index("lag_days").sort_index()

    @staticmethod
    def _season_year(index: pd.DatetimeIndex) -> pd.Index:
        return pd.Index(index.year + (index.month == 12).astype(int), name="year")

    @cached_property
    def seasonal_yearly_mean_difference(self) -> pd.DataFrame:
        frame = self.daily_lag_analysis_frame.copy()
        frame["season"] = frame.index.month.map(self.MONTH_TO_SEASON)  # pyright: ignore
        frame["year"] = self._season_year(frame.index)
        frame["day_type"] = "All Days"
        all_days = (
            frame.groupby(["season", "year", "day_type"])["skt_minus_t2m"]
            .mean()
            .reset_index(name="mean_skt_minus_t2m")
        )

        classified = frame.copy()
        classified["day_type"] = np.where(classified["heatwave"] == 1, "Heatwave Days", "Normal Days")
        classified_days = (
            classified.groupby(["season", "year", "day_type"])["skt_minus_t2m"]
            .mean()
            .reset_index(name="mean_skt_minus_t2m")
        )

        summary = pd.concat([all_days, classified_days], ignore_index=True)
        summary["season"] = pd.Categorical(
            summary["season"],
            categories=["Winter", "Spring", "Summer", "Fall"],
            ordered=True,
        )
        summary["day_type"] = pd.Categorical(
            summary["day_type"],
            categories=["All Days", "Heatwave Days", "Normal Days"],
            ordered=True,
        )
        return summary.sort_values(["season", "year", "day_type"]).reset_index(drop=True)

    def seasonal_trend_summary(self) -> pd.DataFrame:
        records: list[dict[str, str | float | int]] = []
        for (season, day_type), group in self.seasonal_yearly_mean_difference.groupby(["season", "day_type"], observed=False):
            clean_group = group.dropna(subset=["mean_skt_minus_t2m"]).copy()
            if len(clean_group) < 2:
                records.append(
                    {
                        "season": str(season),
                        "day_type": str(day_type),
                        "slope_per_year": float("nan"),
                        "intercept": float("nan"),
                        "r_squared": float("nan"),
                        "slope_p_value": float("nan"),
                        "trend_function": "",
                        "start_year": int(clean_group["year"].min()) if not clean_group.empty else -1,
                        "end_year": int(clean_group["year"].max()) if not clean_group.empty else -1,
                        "n_years": len(clean_group),
                    }
                )
                continue

            years = clean_group["year"].to_numpy(dtype=float)
            values = clean_group["mean_skt_minus_t2m"].to_numpy(dtype=float)
            regression = linregress(years, values)
            slope = float(regression.slope)
            intercept = float(regression.intercept)
            fitted = slope * years + intercept
            ss_res = float(np.sum((values - fitted) ** 2))
            ss_tot = float(np.sum((values - values.mean()) ** 2))
            r_squared = float("nan") if math.isclose(ss_tot, 0.0) else 1 - ss_res / ss_tot
            records.append(
                {
                    "season": str(season),
                    "day_type": str(day_type),
                    "slope_per_year": slope,
                    "intercept": intercept,
                    "r_squared": r_squared,
                    "slope_p_value": float(regression.pvalue),
                    "trend_function": f"y = {slope:.5f} * year + {intercept:.2f}",
                    "start_year": int(years.min()),
                    "end_year": int(years.max()),
                    "n_years": len(clean_group),
                }
            )

        summary = pd.DataFrame(records)
        summary["season"] = pd.Categorical(
            summary["season"],
            categories=["Winter", "Spring", "Summer", "Fall"],
            ordered=True,
        )
        summary["day_type"] = pd.Categorical(
            summary["day_type"],
            categories=["All Days", "Heatwave Days", "Normal Days"],
            ordered=True,
        )
        return summary.sort_values(["season", "day_type"]).reset_index(drop=True)

    @cached_property
    def regional_yearly_mean_difference(self) -> pd.DataFrame:
        records: list[dict[str, str | int | float]] = []
        for region in self.REGION_ORDER:
            region_mask = self.region_masks[region]
            regional_mean_difference = self.skin_air_difference.where(region_mask).mean(
                dim=("latitude", "longitude"),
                skipna=True,
            )
            difference_series: pd.Series = cast(pd.Series, regional_mean_difference.to_pandas())
            difference_series.index = pd.to_datetime(difference_series.index)
            difference_series = difference_series.sort_index()

            regional_extreme = self.heatwave_analysis.extreme_mask.where(region_mask)
            extreme_count = regional_extreme.sum(dim=("latitude", "longitude"), skipna=True)
            land_count = region_mask.sum(dim=("latitude", "longitude"))
            regional_fraction = extreme_count / land_count
            regional_heatwave_mask = cast(pd.Series, regional_fraction.to_pandas()).fillna(0).astype(float)
            regional_heatwave_mask.index = pd.to_datetime(regional_heatwave_mask.index)

            frame = pd.DataFrame(
                {
                    "skt_minus_t2m": difference_series.reindex(regional_heatwave_mask.index),
                    "heatwave": regional_heatwave_mask >= self.heatwave_analysis.min_area_fraction,
                },
                index=regional_heatwave_mask.index,
            ).dropna()
            frame["year"] = frame.index.year
            frame["region"] = region
            frame["day_type"] = "All Days"
            all_days = (
                frame.groupby(["region", "year", "day_type"])["skt_minus_t2m"]
                .mean()
                .reset_index(name="mean_skt_minus_t2m")
            )

            classified = frame.copy()
            classified["day_type"] = np.where(classified["heatwave"], "Heatwave Days", "Normal Days")
            classified_days = (
                classified.groupby(["region", "year", "day_type"])["skt_minus_t2m"]
                .mean()
                .reset_index(name="mean_skt_minus_t2m")
            )
            records.extend(pd.concat([all_days, classified_days], ignore_index=True).to_dict(orient="records"))

        summary = pd.DataFrame(records)
        summary["region"] = pd.Categorical(summary["region"], categories=self.REGION_ORDER, ordered=True)
        summary["day_type"] = pd.Categorical(
            summary["day_type"],
            categories=["All Days", "Heatwave Days", "Normal Days"],
            ordered=True,
        )
        return summary.sort_values(["region", "year", "day_type"]).reset_index(drop=True)

    def regional_trend_summary(self) -> pd.DataFrame:
        records: list[dict[str, str | float | int]] = []
        for (region, day_type), group in self.regional_yearly_mean_difference.groupby(["region", "day_type"], observed=False):
            clean_group = group.dropna(subset=["mean_skt_minus_t2m"]).copy()
            if len(clean_group) < 2:
                records.append(
                    {
                        "region": str(region),
                        "day_type": str(day_type),
                        "slope_per_year": float("nan"),
                        "intercept": float("nan"),
                        "r_squared": float("nan"),
                        "slope_p_value": float("nan"),
                        "trend_function": "",
                        "start_year": int(clean_group["year"].min()) if not clean_group.empty else -1,
                        "end_year": int(clean_group["year"].max()) if not clean_group.empty else -1,
                        "n_years": len(clean_group),
                    }
                )
                continue

            years = clean_group["year"].to_numpy(dtype=float)
            values = clean_group["mean_skt_minus_t2m"].to_numpy(dtype=float)
            regression = linregress(years, values)
            slope = float(regression.slope)
            intercept = float(regression.intercept)
            fitted = slope * years + intercept
            ss_res = float(np.sum((values - fitted) ** 2))
            ss_tot = float(np.sum((values - values.mean()) ** 2))
            r_squared = float("nan") if math.isclose(ss_tot, 0.0) else 1 - ss_res / ss_tot
            records.append(
                {
                    "region": str(region),
                    "day_type": str(day_type),
                    "slope_per_year": slope,
                    "intercept": intercept,
                    "r_squared": r_squared,
                    "slope_p_value": float(regression.pvalue),
                    "trend_function": f"y = {slope:.5f} * year + {intercept:.2f}",
                    "start_year": int(years.min()),
                    "end_year": int(years.max()),
                    "n_years": len(clean_group),
                }
            )

        summary = pd.DataFrame(records)
        summary["region"] = pd.Categorical(summary["region"], categories=self.REGION_ORDER, ordered=True)
        summary["day_type"] = pd.Categorical(
            summary["day_type"],
            categories=["All Days", "Heatwave Days", "Normal Days"],
            ordered=True,
        )
        return summary.sort_values(["region", "day_type"]).reset_index(drop=True)

    @classmethod
    def _bootstrap_difference_in_means(
        cls,
        heatwave_values: pd.Series,
        normal_values: pd.Series,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95,
    ) -> dict[str, float | int | bool]:
        heatwave = heatwave_values.dropna().to_numpy(dtype=float)
        normal = normal_values.dropna().to_numpy(dtype=float)
        if len(heatwave) == 0 or len(normal) == 0:
            return {
                "heatwave_mean": float("nan"),
                "normal_mean": float("nan"),
                "mean_difference": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "significant": False,
                "heatwave_count": len(heatwave),
                "normal_count": len(normal),
            }

        observed_difference = float(heatwave.mean() - normal.mean())
        rng = np.random.default_rng(cls.BOOTSTRAP_SEED)
        heatwave_samples = rng.choice(heatwave, size=(n_bootstrap, len(heatwave)), replace=True)
        normal_samples = rng.choice(normal, size=(n_bootstrap, len(normal)), replace=True)
        bootstrap_differences = heatwave_samples.mean(axis=1) - normal_samples.mean(axis=1)
        alpha = 1.0 - confidence_level
        ci_lower, ci_upper = np.quantile(bootstrap_differences, [alpha / 2, 1 - alpha / 2])
        significant = bool((ci_lower > 0) or (ci_upper < 0))
        return {
            "heatwave_mean": float(heatwave.mean()),
            "normal_mean": float(normal.mean()),
            "mean_difference": observed_difference,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "significant": significant,
            "heatwave_count": len(heatwave),
            "normal_count": len(normal),
        }

    def _bootstrap_summary_frame(
        self,
        grouped_differences: dict[str, tuple[pd.Series, pd.Series]],
        *,
        group_name: str,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        records: list[dict[str, str | float | int | bool]] = []
        for label, (heatwave_values, normal_values) in grouped_differences.items():
            summary = self._bootstrap_difference_in_means(
                heatwave_values=heatwave_values,
                normal_values=normal_values,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
            )
            records.append(
                {
                    group_name: label,
                    **summary,
                }
            )
        return pd.DataFrame(records)

    def bootstrap_significance_by_season(
        self,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        summary = self._bootstrap_summary_frame(
            self.seasonal_heatwave_normal_difference,
            group_name="season",
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        return summary.sort_values("season").reset_index(drop=True)

    def bootstrap_significance_by_region(
        self,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        summary = self._bootstrap_summary_frame(
            self.regional_heatwave_normal_difference,
            group_name="region",
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        summary["region"] = pd.Categorical(summary["region"], categories=self.REGION_ORDER, ordered=True)
        return summary.sort_values("region").reset_index(drop=True)

    def bootstrap_significance_by_decade_and_season(
        self,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        records: list[dict[str, str | float | int | bool]] = []
        for decade, seasonal_differences in self.decadal_seasonal_heatwave_normal_difference.items():
            seasonal_summary = self._bootstrap_summary_frame(
                seasonal_differences,
                group_name="season",
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
            )
            for record in seasonal_summary.to_dict(orient="records"):
                records.append(
                    {
                        "decade": decade,
                        **record,
                    }
                )
        summary = pd.DataFrame(records)
        summary["season"] = pd.Categorical(
            summary["season"],
            categories=["Winter", "Spring", "Summer", "Fall"],
            ordered=True,
        )
        return summary.sort_values(["decade", "season"]).reset_index(drop=True)

    def export_bootstrap_significance_tables(
        self,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95,
    ) -> None:
        self.bootstrap_significance_by_season(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        ).to_csv("skt_minus_t2m_bootstrap_significance_by_season.csv", index=False)
        self.bootstrap_significance_by_region(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        ).to_csv("skt_minus_t2m_bootstrap_significance_by_region.csv", index=False)
        self.bootstrap_significance_by_decade_and_season(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        ).to_csv("skt_minus_t2m_bootstrap_significance_by_decade_and_season.csv", index=False)

    def export_seasonal_trend_tables(self) -> None:
        self.seasonal_yearly_mean_difference.to_csv(
            "skt_minus_t2m_seasonal_yearly_means.csv",
            index=False,
        )
        self.seasonal_trend_summary().to_csv(
            "skt_minus_t2m_seasonal_trend_summary.csv",
            index=False,
        )
        self.regional_yearly_mean_difference.to_csv(
            "skt_minus_t2m_regional_yearly_means.csv",
            index=False,
        )
        self.regional_trend_summary().to_csv(
            "skt_minus_t2m_regional_trend_summary.csv",
            index=False,
        )

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

    def plot_distribution_by_season(self) -> None:
        seasonal_differences = self.seasonal_heatwave_normal_difference
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        all_values = [
            series.to_numpy()
            for seasonal_pair in seasonal_differences.values()
            for series in seasonal_pair
            if not series.empty
        ]
        if not all_values:
            raise ValueError("No SKT - T2M values available for seasonal plotting.")
        combined_values = np.concatenate(all_values)
        x_min = float(combined_values.min())
        x_max = float(combined_values.max())
        if np.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        x_grid = np.linspace(x_min, x_max, 400)
        bins = np.linspace(x_min, x_max, 30)

        for ax, (season, (heatwave_difference, normal_difference)) in zip(axes_flat, seasonal_differences.items()):
            if normal_difference.empty or heatwave_difference.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                ax.set_title(season, fontsize=15)
                ax.grid(True, alpha=0.3)
                continue

            normal_values = normal_difference.to_numpy()
            heatwave_values = heatwave_difference.to_numpy()
            ax.hist(
                normal_values,
                bins=bins,  # pyright: ignore
                density=True,
                color="steelblue",
                alpha=0.25,
                label="Normal Days",
            )
            ax.hist(
                heatwave_values,
                bins=bins,  # pyright: ignore
                density=True,
                color="firebrick",
                alpha=0.25,
                label="Heatwave Days",
            )

            if len(normal_values) > 1:
                normal_kde = gaussian_kde(normal_values)
                ax.plot(x_grid, normal_kde(x_grid), color="steelblue", linewidth=2)
                ax.fill_between(x_grid, normal_kde(x_grid), color="steelblue", alpha=0.2)
            if len(heatwave_values) > 1:
                heatwave_kde = gaussian_kde(heatwave_values)
                ax.plot(x_grid, heatwave_kde(x_grid), color="firebrick", linewidth=2)
                ax.fill_between(x_grid, heatwave_kde(x_grid), color="firebrick", alpha=0.2)

            ax.axvline(float(normal_difference.mean()), color="steelblue", linestyle="--", linewidth=1.5)
            ax.axvline(float(heatwave_difference.mean()), color="firebrick", linestyle="--", linewidth=1.5)
            ax.set_title(season, fontsize=15)
            ax.grid(True, alpha=0.3)

        for ax in axes[1, :]:
            ax.set_xlabel("SKT - T2M", fontsize=13)
        for ax in axes[:, 0]:
            ax.set_ylabel("Density", fontsize=13)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, ncol=2, fontsize=12, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig("skt_minus_t2m_distribution_heatwave_vs_normal_by_season.png", dpi=300)

    def plot_distribution_by_region(self) -> None:
        regional_differences = self.regional_heatwave_normal_difference
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        all_values = [
            series.to_numpy()
            for regional_pair in regional_differences.values()
            for series in regional_pair
            if not series.empty
        ]
        if not all_values:
            raise ValueError("No SKT - T2M values available for regional plotting.")
        combined_values = np.concatenate(all_values)
        x_min = float(combined_values.min())
        x_max = float(combined_values.max())
        if np.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        x_grid = np.linspace(x_min, x_max, 400)
        bins = np.linspace(x_min, x_max, 30)

        for ax, (region, (heatwave_difference, normal_difference)) in zip(axes_flat, regional_differences.items()):
            if normal_difference.empty or heatwave_difference.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                ax.set_title(region, fontsize=15)
                ax.grid(True, alpha=0.3)
                continue

            normal_values = normal_difference.to_numpy()
            heatwave_values = heatwave_difference.to_numpy()
            ax.hist(
                normal_values,
                bins=bins,  # pyright: ignore
                density=True,
                color="steelblue",
                alpha=0.25,
                label="Normal Days",
            )
            ax.hist(
                heatwave_values,
                bins=bins,  # pyright: ignore
                density=True,
                color="firebrick",
                alpha=0.25,
                label="Heatwave Days",
            )

            if len(normal_values) > 1:
                normal_kde = gaussian_kde(normal_values)
                ax.plot(x_grid, normal_kde(x_grid), color="steelblue", linewidth=2)
                ax.fill_between(x_grid, normal_kde(x_grid), color="steelblue", alpha=0.2)
            if len(heatwave_values) > 1:
                heatwave_kde = gaussian_kde(heatwave_values)
                ax.plot(x_grid, heatwave_kde(x_grid), color="firebrick", linewidth=2)
                ax.fill_between(x_grid, heatwave_kde(x_grid), color="firebrick", alpha=0.2)

            ax.axvline(float(normal_difference.mean()), color="steelblue", linestyle="--", linewidth=1.5)
            ax.axvline(float(heatwave_difference.mean()), color="firebrick", linestyle="--", linewidth=1.5)
            ax.set_title(region, fontsize=15)
            ax.grid(True, alpha=0.3)

        for ax in axes[1, :]:
            ax.set_xlabel("SKT - T2M", fontsize=13)
        for ax in axes[:, 0]:
            ax.set_ylabel("Density", fontsize=13)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, ncol=2, fontsize=12, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig("skt_minus_t2m_distribution_heatwave_vs_normal_by_region.png", dpi=300)

    def plot_distribution_by_season_and_region(self) -> None:
        seasons = ("Winter", "Spring", "Summer", "Fall")
        regions = self.REGION_ORDER
        regional_differences = self.regional_heatwave_normal_difference
        season_region_differences: dict[str, dict[str, tuple[pd.Series, pd.Series]]] = {}
        all_values: list[np.ndarray] = []

        for season in seasons:
            season_months = [
                month for month, mapped_season in self.MONTH_TO_SEASON.items()
                if mapped_season == season
            ]
            season_region_differences[season] = {}
            for region in regions:
                heatwave_difference, normal_difference = regional_differences[region]
                seasonal_heatwave = heatwave_difference.loc[heatwave_difference.index.month.isin(season_months)]  # pyright: ignore
                seasonal_normal = normal_difference.loc[normal_difference.index.month.isin(season_months)]  # pyright: ignore
                season_region_differences[season][region] = (seasonal_heatwave, seasonal_normal)
                if not seasonal_heatwave.empty:
                    all_values.append(seasonal_heatwave.to_numpy())
                if not seasonal_normal.empty:
                    all_values.append(seasonal_normal.to_numpy())

        if not all_values:
            raise ValueError("No SKT - T2M values available for season-region plotting.")

        combined_values = np.concatenate(all_values)
        x_min = float(combined_values.min())
        x_max = float(combined_values.max())
        if np.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        x_grid = np.linspace(x_min, x_max, 400)
        bins = np.linspace(x_min, x_max, 30)

        fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True, sharey=True)
        for row_index, season in enumerate(seasons):
            for col_index, region in enumerate(regions):
                ax = axes[row_index, col_index]
                heatwave_difference, normal_difference = season_region_differences[season][region]
                if normal_difference.empty or heatwave_difference.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "Insufficient data",
                        ha="center",
                        va="center",
                        fontsize=11,
                        transform=ax.transAxes,
                    )
                    ax.grid(True, alpha=0.3)
                else:
                    normal_values = normal_difference.to_numpy()
                    heatwave_values = heatwave_difference.to_numpy()
                    ax.hist(
                        normal_values,
                        bins=bins,  # pyright: ignore
                        density=True,
                        color="steelblue",
                        alpha=0.25,
                        label="Normal Days",
                    )
                    ax.hist(
                        heatwave_values,
                        bins=bins,  # pyright: ignore
                        density=True,
                        color="firebrick",
                        alpha=0.25,
                        label="Heatwave Days",
                    )
                    if len(normal_values) > 1:
                        normal_kde = gaussian_kde(normal_values)
                        ax.plot(x_grid, normal_kde(x_grid), color="steelblue", linewidth=1.6)
                        ax.fill_between(x_grid, normal_kde(x_grid), color="steelblue", alpha=0.2)
                    if len(heatwave_values) > 1:
                        heatwave_kde = gaussian_kde(heatwave_values)
                        ax.plot(x_grid, heatwave_kde(x_grid), color="firebrick", linewidth=1.6)
                        ax.fill_between(x_grid, heatwave_kde(x_grid), color="firebrick", alpha=0.2)
                    ax.axvline(float(normal_difference.mean()), color="steelblue", linestyle="--", linewidth=1.2)
                    ax.axvline(float(heatwave_difference.mean()), color="firebrick", linestyle="--", linewidth=1.2)
                    ax.grid(True, alpha=0.3)

                if row_index == 0:
                    ax.set_title(region, fontsize=15)
                if col_index == 0:
                    ax.set_ylabel(f"{season}\nDensity", fontsize=13)
                if row_index == len(seasons) - 1:
                    ax.set_xlabel("SKT - T2M", fontsize=13)
                ax.tick_params(axis="both", labelsize=10)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig("skt_minus_t2m_distribution_heatwave_vs_normal_by_season_and_region.png", dpi=300)

    def plot_lagged_relationships(self, max_lag: int = 7) -> None:
        lagged_relationships = self.lagged_relationships(max_lag=max_lag)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(
            lagged_relationships.index.to_numpy(),
            lagged_relationships["correlation"].to_numpy(),
            color="black",
            marker="o",
            linewidth=1.8,
        )
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[0].axvline(0, color="gray", linestyle=":", linewidth=1)
        axes[0].set_ylabel("Correlation", fontsize=13)
        axes[0].set_title("Lagged Relationship Between SKT - T2M and Heatwaves", fontsize=16)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            lagged_relationships.index.to_numpy(),
            lagged_relationships["heatwave_rate_low_skt_minus_t2m"].to_numpy(),
            color="steelblue",
            marker="o",
            linewidth=1.8,
            label="Low SKT - T2M days (bottom quartile)",
        )
        axes[1].plot(
            lagged_relationships.index.to_numpy(),
            lagged_relationships["heatwave_rate_high_skt_minus_t2m"].to_numpy(),
            color="firebrick",
            marker="o",
            linewidth=1.8,
            label="High SKT - T2M days (top quartile)",
        )
        axes[1].axvline(0, color="gray", linestyle=":", linewidth=1)
        axes[1].set_xlabel(
            "Lag (days): positive means SKT - T2M leads future heatwave conditions",
            fontsize=13,
        )
        axes[1].set_ylabel("Heatwave Probability", fontsize=13)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=11)

        fig.tight_layout()
        fig.savefig("skt_minus_t2m_lagged_heatwave_relationships.png", dpi=300)

    def plot_seasonal_trends(self) -> None:
        yearly_means = self.seasonal_yearly_mean_difference
        trend_summary = self.seasonal_trend_summary()
        season_order = ("Winter", "Spring", "Summer", "Fall")
        annotation_position_by_season = {
            "Winter": {"x": 0.02, "y": 0.98, "ha": "left", "va": "top"},
            "Spring": {"x": 0.02, "y": 0.98, "ha": "left", "va": "top"},
            "Summer": {"x": 0.02, "y": 0.02, "ha": "left", "va": "bottom"},
            "Fall": {"x": 0.02, "y": 0.02, "ha": "left", "va": "bottom"},
        }
        style_by_day_type = {
            "All Days": {"color": "black", "marker": "o"},
            "Heatwave Days": {"color": "firebrick", "marker": "^"},
            "Normal Days": {"color": "steelblue", "marker": "s"},
        }

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        for ax, season in zip(axes_flat, season_order):
            season_means = yearly_means.loc[yearly_means["season"] == season]
            season_trends = trend_summary.loc[trend_summary["season"] == season]
            annotation_lines: list[str] = []
            for day_type in ("All Days", "Heatwave Days", "Normal Days"):
                group = season_means.loc[season_means["day_type"] == day_type].dropna(subset=["mean_skt_minus_t2m"])
                if group.empty:
                    continue
                style = style_by_day_type[day_type]
                years = group["year"].to_numpy(dtype=float)
                values = group["mean_skt_minus_t2m"].to_numpy(dtype=float)
                ax.plot(
                    years,
                    values,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.5,
                    markersize=4,
                    label=day_type,
                )

                trend_row = season_trends.loc[season_trends["day_type"] == day_type]
                if trend_row.empty or pd.isna(trend_row["slope_per_year"].iloc[0]):
                    continue
                slope = float(trend_row["slope_per_year"].iloc[0])
                intercept = float(trend_row["intercept"].iloc[0])
                p_value = float(trend_row["slope_p_value"].iloc[0])
                fitted = slope * years + intercept
                ax.plot(
                    years,
                    fitted,
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                )
                short_day_type = day_type.replace(" Days", "")
                intercept_text = f"+ {abs(intercept):.2f}" if intercept >= 0 else f"- {abs(intercept):.2f}"
                annotation_lines.append(
                    f"{short_day_type}: y={slope:.4f}x {intercept_text}, p={p_value:.3g}"
                )

            ax.set_title(season, fontsize=18)
            ax.grid(True, alpha=0.3)
            if annotation_lines:
                annotation_position = annotation_position_by_season[season]
                ax.text(
                    annotation_position["x"],
                    annotation_position["y"],
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    ha=annotation_position["ha"],
                    va=annotation_position["va"],
                    fontsize=10.5,
                    bbox={
                        "boxstyle": "round",
                        "facecolor": "white",
                        "edgecolor": "lightgray",
                        "alpha": 0.85,
                    },
                )

        for ax in axes[1, :]:
            ax.set_xlabel("Year", fontsize=15)
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean SKT - T2M", fontsize=15)
        for ax in axes_flat:
            ax.tick_params(axis="both", labelsize=12)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=13, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig("skt_minus_t2m_seasonal_trends.png", dpi=300)

    def plot_regional_trends(self) -> None:
        yearly_means = self.regional_yearly_mean_difference
        trend_summary = self.regional_trend_summary()
        style_by_day_type = {
            "All Days": {"color": "black", "marker": "o"},
            "Heatwave Days": {"color": "firebrick", "marker": "^"},
            "Normal Days": {"color": "steelblue", "marker": "s"},
        }

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        for ax, region in zip(axes_flat, self.REGION_ORDER):
            region_means = yearly_means.loc[yearly_means["region"] == region]
            region_trends = trend_summary.loc[trend_summary["region"] == region]
            annotation_lines: list[str] = []
            for day_type in ("All Days", "Heatwave Days", "Normal Days"):
                group = region_means.loc[region_means["day_type"] == day_type].dropna(subset=["mean_skt_minus_t2m"])
                if group.empty:
                    continue
                style = style_by_day_type[day_type]
                years = group["year"].to_numpy(dtype=float)
                values = group["mean_skt_minus_t2m"].to_numpy(dtype=float)
                ax.plot(
                    years,
                    values,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.5,
                    markersize=4,
                    label=day_type,
                )

                trend_row = region_trends.loc[region_trends["day_type"] == day_type]
                if trend_row.empty or pd.isna(trend_row["slope_per_year"].iloc[0]):
                    continue
                slope = float(trend_row["slope_per_year"].iloc[0])
                intercept = float(trend_row["intercept"].iloc[0])
                p_value = float(trend_row["slope_p_value"].iloc[0])
                fitted = slope * years + intercept
                ax.plot(
                    years,
                    fitted,
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                )
                short_day_type = day_type.replace(" Days", "")
                intercept_text = f"+ {abs(intercept):.2f}" if intercept >= 0 else f"- {abs(intercept):.2f}"
                annotation_lines.append(
                    f"{short_day_type}: y={slope:.4f}x {intercept_text}, p={p_value:.3g}"
                )

            ax.set_title(region, fontsize=18)
            ax.grid(True, alpha=0.3)
            if annotation_lines:
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10.5,
                    bbox={
                        "boxstyle": "round",
                        "facecolor": "white",
                        "edgecolor": "lightgray",
                        "alpha": 0.85,
                    },
                )

        for ax in axes[1, :]:
            ax.set_xlabel("Year", fontsize=15)
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean SKT - T2M", fontsize=15)
        for ax in axes_flat:
            ax.tick_params(axis="both", labelsize=12)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=13, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig("skt_minus_t2m_regional_trends.png", dpi=300)


if __name__ == "__main__":

    skt_reader = Era5TemperatureReader(root_dir=DATA_ROOT, var_name="skt", from_year=2025, to_year=2025)
    t2m_reader = Era5TemperatureReader(root_dir=DATA_ROOT, var_name="t2m", from_year=2025, to_year=2025)
    daily_mean = DailyMean(skt_reader=skt_reader, t2m_reader=t2m_reader)
    daily_mean.plot()

    from_year: int = 2000
    to_year: int = 2025
    skt_reader = Era5TemperatureReader(root_dir=DATA_ROOT, var_name="skt", from_year=from_year, to_year=to_year)
    t2m_reader = Era5TemperatureReader(root_dir=DATA_ROOT, var_name="t2m", from_year=from_year, to_year=to_year)

    persistence = TemperaturePersistenceAnalysis(skt_reader=skt_reader, t2m_reader=t2m_reader)
    persistence.plot_autocorrelation(max_lag=30)

    skt_heatwave = SameTimeHeatwaveAnalysis(var_name="skt", from_year=from_year, to_year=to_year)
    t2m_heatwave = SameTimeHeatwaveAnalysis(var_name="t2m", from_year=from_year, to_year=to_year)

    skt_heatwave.plot_frequency_by_year()
    t2m_heatwave.plot_frequency_by_year()

    skt_t2m_difference = SkinAirTemperatureDifferenceAnalysis(
        skt_reader=skt_reader,
        t2m_reader=t2m_reader,
        heatwave_analysis=skt_heatwave,
    )
    skt_t2m_difference.plot_distribution()
    skt_t2m_difference.plot_distribution_by_season()
    skt_t2m_difference.plot_distribution_by_region()
    skt_t2m_difference.plot_distribution_by_season_and_region()
    skt_t2m_difference.plot_lagged_relationships(max_lag=7)
    skt_t2m_difference.plot_seasonal_trends()
    skt_t2m_difference.plot_regional_trends()
    skt_t2m_difference.export_bootstrap_significance_tables(n_bootstrap=5000, confidence_level=0.95)
    skt_t2m_difference.export_seasonal_trend_tables()

    for year in range(from_year, to_year + 1):
        skt_heatwave.plot_count_map(year)
        t2m_heatwave.plot_count_map(year)

    skt_heatwave.plot_heatwave_distribution()
    t2m_heatwave.plot_heatwave_distribution()
