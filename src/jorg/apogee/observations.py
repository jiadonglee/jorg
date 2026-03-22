"""APOGEE DR17 download, selection, and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Iterable
import urllib.parse
import urllib.request

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .constants import apogee_wavelength_grid
from .contracts import ObservationDatasetMetadata, save_observation_dataset

DEFAULT_ALLSTARLITE_URL = (
    "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
    "allStarLite-dr17-synspec_rev1.fits"
)
DEFAULT_ALLSTAR_URL = (
    "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
    "allStar-dr17-synspec_rev1.fits"
)
DEFAULT_APSTAR_BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"
DEFAULT_ASPCAPSTAR_BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1"


@dataclass(frozen=True)
class ApogeeObservationConfig:
    """Configuration for building APOGEE quick/formal benchmark sets."""

    product: str
    selection_size: int
    teff_range: tuple[float, float] = (3000.0, 4500.0)
    min_snr: float = 80.0
    require_starflag_zero: bool = True
    require_aspcapflag_zero: bool = True
    percentile: float = 95.0
    continuum_window: int = 64
    knot_stride: int = 96
    allstar_url: str = DEFAULT_ALLSTARLITE_URL
    apstar_base_url: str = DEFAULT_APSTAR_BASE
    aspcapstar_base_url: str = DEFAULT_ASPCAPSTAR_BASE


def download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    """Download one APOGEE product file if it is not already cached."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def load_allstar_table(path_or_url: str | Path, cache_dir: Path | None = None) -> pd.DataFrame:
    """Load an allStar/allStarLite FITS table into a pandas DataFrame."""
    local_path = Path(path_or_url)
    if urllib.parse.urlparse(str(path_or_url)).scheme in {"http", "https"}:
        if cache_dir is None:
            raise ValueError("cache_dir must be provided when loading a remote allStar table.")
        local_path = download_file(str(path_or_url), cache_dir / Path(str(path_or_url)).name)
    with fits.open(local_path, memmap=True) as hdul:
        table_hdu = next(hdu for hdu in hdul if getattr(hdu, "data", None) is not None)
        data = table_hdu.data
        df = pd.DataFrame(np.array(data).byteswap().newbyteorder())
    for column in df.columns:
        if df[column].dtype.kind in {"S", "O"}:
            df[column] = df[column].map(_decode_value)
    return df


def _decode_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore").strip()
    return value


def filter_quick_benchmark_candidates(
    allstar_df: pd.DataFrame,
    *,
    teff_range: tuple[float, float] = (3000.0, 4500.0),
    min_snr: float = 80.0,
    require_starflag_zero: bool = True,
    require_aspcapflag_zero: bool = True,
) -> pd.DataFrame:
    """Apply the planned APOGEE quality filters to allStar/allStarLite rows."""
    df = allstar_df.copy()
    if "TEFF" not in df.columns:
        raise KeyError("Expected allStar table to include TEFF.")
    mask = (df["TEFF"] >= teff_range[0]) & (df["TEFF"] <= teff_range[1])
    if "SNR" in df.columns:
        mask &= df["SNR"] >= min_snr
    if require_aspcapflag_zero and "ASPCAPFLAG" in df.columns:
        mask &= df["ASPCAPFLAG"].fillna(0).astype(np.int64) == 0
    if require_starflag_zero and "STARFLAG" in df.columns:
        mask &= df["STARFLAG"].fillna(0).astype(np.int64) == 0
    return df.loc[mask].reset_index(drop=True)


def select_covering_subset(
    candidate_df: pd.DataFrame,
    n_select: int,
    *,
    feature_columns: tuple[str, ...] = ("TEFF", "LOGG", "M_H"),
) -> pd.DataFrame:
    """Select a parameter-space-covering subset with greedy farthest-point sampling."""
    if len(candidate_df) <= n_select:
        return candidate_df.copy()

    features = candidate_df.loc[:, feature_columns].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(features).all(axis=1)
    valid_df = candidate_df.loc[finite_mask].reset_index(drop=True)
    features = features[finite_mask]
    mins = features.min(axis=0)
    spans = np.where(features.max(axis=0) > mins, features.max(axis=0) - mins, 1.0)
    scaled = (features - mins) / spans

    centroid = scaled.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(scaled - centroid[None, :], axis=1)))]
    min_distance = np.linalg.norm(scaled - scaled[selected[0]][None, :], axis=1)
    while len(selected) < n_select:
        next_index = int(np.argmax(min_distance))
        if next_index in selected:
            break
        selected.append(next_index)
        candidate_distance = np.linalg.norm(scaled - scaled[next_index][None, :], axis=1)
        min_distance = np.minimum(min_distance, candidate_distance)
    return valid_df.iloc[selected].reset_index(drop=True)


def split_apogee_chips(wavelengths: np.ndarray, gap_factor: float = 5.0) -> list[slice]:
    """Split a wavelength array into APOGEE chips by large wavelength gaps."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError("wavelengths must be 1-D")
    diff = np.diff(wl)
    positive = diff[np.isfinite(diff) & (diff > 0)]
    if positive.size == 0:
        return [slice(0, wl.size)]
    threshold = float(np.median(positive) * gap_factor)
    boundaries = np.where(diff > threshold)[0]
    start = 0
    slices: list[slice] = []
    for boundary in boundaries:
        slices.append(slice(start, boundary + 1))
        start = boundary + 1
    slices.append(slice(start, wl.size))
    return slices


def pseudo_continuum_normalize(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray | None = None,
    *,
    percentile: float = 95.0,
    window: int = 64,
    knot_stride: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one APOGEE spectrum chip-by-chip with a spline pseudo-continuum."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if ivar is None:
        valid = np.isfinite(flux) & (flux > 0)
    else:
        valid = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (flux > 0)

    continuum = np.full_like(flux, np.nan, dtype=np.float64)
    chips = split_apogee_chips(wl)
    for chip in chips:
        chip_flux = flux[chip]
        chip_wl = wl[chip]
        chip_valid = valid[chip]
        chip_cont = _fit_chip_continuum(
            chip_wl,
            chip_flux,
            chip_valid,
            percentile=percentile,
            window=window,
            knot_stride=knot_stride,
        )
        continuum[chip] = chip_cont

    normalized = np.divide(
        flux,
        continuum,
        out=np.full_like(flux, np.nan, dtype=np.float64),
        where=np.isfinite(continuum) & (continuum > 0),
    )
    return normalized.astype(np.float32), continuum.astype(np.float32)


def _fit_chip_continuum(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    valid: np.ndarray,
    *,
    percentile: float,
    window: int,
    knot_stride: int,
) -> np.ndarray:
    anchor_x: list[float] = []
    anchor_y: list[float] = []
    n_pix = len(wavelengths)
    for start in range(0, n_pix, window):
        stop = min(start + window, n_pix)
        mask = valid[start:stop]
        if not np.any(mask):
            continue
        chip_flux = flux[start:stop][mask]
        chip_wl = wavelengths[start:stop][mask]
        anchor_x.append(float(np.median(chip_wl)))
        anchor_y.append(float(np.percentile(chip_flux, percentile)))

    if len(anchor_x) < 4:
        fallback = np.nanmedian(flux[valid]) if np.any(valid) else 1.0
        return np.full_like(flux, fallback, dtype=np.float64)

    anchor_x_arr = np.asarray(anchor_x, dtype=np.float64)
    anchor_y_arr = np.asarray(anchor_y, dtype=np.float64)
    order = np.argsort(anchor_x_arr)
    anchor_x_arr = anchor_x_arr[order]
    anchor_y_arr = anchor_y_arr[order]

    reduced_x = anchor_x_arr[:: max(1, knot_stride // max(window, 1))]
    reduced_y = anchor_y_arr[:: max(1, knot_stride // max(window, 1))]
    if reduced_x[-1] != anchor_x_arr[-1]:
        reduced_x = np.append(reduced_x, anchor_x_arr[-1])
        reduced_y = np.append(reduced_y, anchor_y_arr[-1])
    if reduced_x[0] != anchor_x_arr[0]:
        reduced_x = np.insert(reduced_x, 0, anchor_x_arr[0])
        reduced_y = np.insert(reduced_y, 0, anchor_y_arr[0])
    if len(reduced_x) < 4:
        reduced_x = anchor_x_arr
        reduced_y = anchor_y_arr

    spline = CubicSpline(reduced_x, reduced_y, bc_type="natural", extrapolate=True)
    continuum = spline(wavelengths)
    positive_floor = np.nanpercentile(reduced_y, 5)
    continuum = np.where(continuum > 0, continuum, positive_floor)
    return continuum


def extract_apogee_spectrum(product_path: Path | str) -> dict[str, np.ndarray]:
    """Read wavelength, flux, ivar, and mask arrays from an APOGEE FITS product."""
    with fits.open(product_path, memmap=True) as hdul:
        flux_hdu = _select_hdu(hdul, preferred_names=("FLUX",), fallback_index=1)
        flux = _collapse_spectrum(flux_hdu.data)
        ivar_hdu = _select_hdu(hdul, preferred_names=("IVAR",), fallback_index=2, allow_missing=True)
        mask_hdu = _select_hdu(
            hdul,
            preferred_names=("MASK", "PIXMASK", "BITMASK"),
            fallback_index=3,
            allow_missing=True,
        )
        wavelength = _extract_wavelength(hdul, flux_hdu.header, flux.shape[-1])
        if ivar_hdu is not None:
            ivar = _collapse_spectrum(ivar_hdu.data)
        else:
            ivar = np.ones_like(flux, dtype=np.float32)
        if mask_hdu is not None:
            mask = _collapse_spectrum(mask_hdu.data).astype(np.int64)
        else:
            mask = np.zeros_like(flux, dtype=np.int64)
    return {
        "wavelength": wavelength.astype(np.float64),
        "flux": flux.astype(np.float32),
        "ivar": ivar.astype(np.float32),
        "mask": mask,
    }


def _select_hdu(
    hdul: fits.HDUList,
    *,
    preferred_names: tuple[str, ...],
    fallback_index: int,
    allow_missing: bool = False,
):
    for hdu in hdul:
        name = (getattr(hdu, "name", "") or "").upper()
        if name in preferred_names:
            return hdu
    if fallback_index < len(hdul) and getattr(hdul[fallback_index], "data", None) is not None:
        return hdul[fallback_index]
    if allow_missing:
        return None
    raise KeyError(f"Could not find HDU {preferred_names} in {hdul.filename()!r}.")


def _collapse_spectrum(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0]
    raise ValueError(f"Unsupported APOGEE spectrum array shape: {arr.shape}")


def _extract_wavelength(hdul: fits.HDUList, header: fits.Header, n_pix: int) -> np.ndarray:
    for hdu in hdul:
        name = (getattr(hdu, "name", "") or "").upper()
        if name in {"WAVE", "WAVELENGTH"} and getattr(hdu, "data", None) is not None:
            return _collapse_spectrum(hdu.data)
    if {"CRVAL1", "CDELT1", "NAXIS1"} <= set(header.keys()):
        crval = float(header["CRVAL1"])
        cdelt = float(header["CDELT1"])
        indices = np.arange(int(header.get("NAXIS1", n_pix)), dtype=np.float64)
        if header.get("DC-FLAG", 0) == 1:
            return np.power(10.0, crval + cdelt * indices)
        return crval + cdelt * indices
    return apogee_wavelength_grid()


def _first_present(row: pd.Series, names: Iterable[str]) -> str | None:
    for name in names:
        if name in row.index:
            value = _decode_value(row[name])
            if value is None:
                continue
            if isinstance(value, float) and not np.isfinite(value):
                continue
            value_str = str(value).strip()
            if value_str:
                return value_str
    return None


def apogee_product_url(
    row: pd.Series,
    *,
    product: str,
    apstar_base_url: str = DEFAULT_APSTAR_BASE,
    aspcapstar_base_url: str = DEFAULT_ASPCAPSTAR_BASE,
) -> str:
    """Build a best-effort DR17 APOGEE product URL for one star."""
    relative_candidates = {
        "apStar": ("APSTAR", "APSTARFILE", "FILE"),
        "aspcapStar": ("ASPCAPSTAR", "ASPCAPSTARFILE", "SPECFILE"),
    }
    rel = _first_present(row, relative_candidates.get(product, ()))
    if rel and rel.endswith(".fits"):
        if rel.startswith("http"):
            return rel
        base = apstar_base_url if product == "apStar" else aspcapstar_base_url
        return urllib.parse.urljoin(base + "/", rel.lstrip("/"))

    telescope = _first_present(row, ("TELESCOPE", "TELESCOPE_ID"))
    field = _first_present(row, ("FIELD", "FIELD_NAME"))
    apogee_id = _first_present(row, ("APOGEE_ID", "APSTAR_ID", "2MASS_ID"))
    if telescope is None or field is None or apogee_id is None:
        raise KeyError(f"Could not construct {product} URL from row columns.")

    if product == "apStar":
        return f"{apstar_base_url}/{telescope}/{field}/apStar-dr17-{apogee_id}.fits"
    if product == "aspcapStar":
        return f"{aspcapstar_base_url}/{telescope}/{field}/aspcapStar-dr17-{apogee_id}.fits"
    raise ValueError(f"Unsupported APOGEE product type: {product}")


def build_observation_benchmark(
    selected_df: pd.DataFrame,
    *,
    output_root: Path | str,
    product: str,
    cache_dir: Path | str,
    normalization: dict[str, Any] | None = None,
    apstar_base_url: str = DEFAULT_APSTAR_BASE,
    aspcapstar_base_url: str = DEFAULT_ASPCAPSTAR_BASE,
) -> Path:
    """Download and normalize a selected APOGEE benchmark sample."""
    normalization = normalization or {"percentile": 95.0, "window": 64, "knot_stride": 96}
    cache_dir = Path(cache_dir)
    output_root = Path(output_root)

    flux_rows: list[np.ndarray] = []
    ivar_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    source_ids: list[str] = []

    for _, row in selected_df.iterrows():
        url = apogee_product_url(
            row,
            product=product,
            apstar_base_url=apstar_base_url,
            aspcapstar_base_url=aspcapstar_base_url,
        )
        local_name = Path(urllib.parse.urlparse(url).path).name
        local_path = download_file(url, cache_dir / product / local_name)
        spectrum = extract_apogee_spectrum(local_path)
        norm_flux, _ = pseudo_continuum_normalize(
            spectrum["wavelength"],
            spectrum["flux"],
            spectrum["ivar"],
            percentile=float(normalization["percentile"]),
            window=int(normalization["window"]),
            knot_stride=int(normalization["knot_stride"]),
        )
        flux_rows.append(norm_flux)
        ivar_rows.append(spectrum["ivar"])
        mask_rows.append(spectrum["mask"] == 0)
        label_rows.append(
            np.array(
                [
                    float(row.get("TEFF", np.nan)),
                    float(row.get("LOGG", np.nan)),
                    float(row.get("M_H", np.nan)),
                ],
                dtype=np.float32,
            )
        )
        source_ids.append(_first_present(row, ("APOGEE_ID", "APSTAR_ID", "2MASS_ID")) or local_name)

    metadata = ObservationDatasetMetadata(
        dataset_name=output_root.name,
        product=product,
        selection={"n_stars": len(source_ids)},
        source_table="selected_rows",
        normalization=normalization,
        notes={},
    )
    save_observation_dataset(
        output_root,
        flux=np.stack(flux_rows, axis=0),
        ivar=np.stack(ivar_rows, axis=0),
        mask=np.stack(mask_rows, axis=0),
        labels=np.stack(label_rows, axis=0),
        source_ids=np.asarray(source_ids),
        metadata=metadata,
    )
    return output_root
