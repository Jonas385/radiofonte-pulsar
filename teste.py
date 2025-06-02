from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from beam_loader import BeamLoaderLIT

from radioskysims.domain.common.models import AngularRange


class Beam:
    def __init__(self, path=None, load=True, observer=None, **kwargs):
        self.datasets = {}
        self.observer = observer
        self.path = None
        self._ds = None
        self.kwargs = kwargs
        if path:
            if isinstance(path, str):
                self.path = Path(path)
            if load:
                self._ds = self._load(path, **self.kwargs)
                self.datasets = self._normalize(self._ds)
        else:
            msg = "Path to beam data file is required."
            raise ValueError(msg)
    
    def _normalize(self, ds):
        """Normalize the amplitude of the dataset."""
        if ds is None:
            msg = "No raw dataset found to normalize."
            raise ValueError(msg)
        ds = self._ds.copy()
        max_vals = ds.amplitude.max(dim='theta')
        ds["amplitude"] = ds.amplitude - max_vals
        ds.attrs['name'] = "normalized"
        return {ds.attrs['name']: ds}
    

    def _load(self, path=None, **kwargs):
        """Load beam data from a file."""
        _bl = BeamLoaderLIT(path, **kwargs)
        return _bl.ds
    
    # fim da ingestao de dados
    def _set_angular_range(self, theta_min: float, theta_max: float, resolution: float, angular_range=None) -> AngularRange | None:
        """Set the angular range for the beam."""
        args = [theta_min, theta_max, resolution]
        if sum(1 for _ in filter(None.__ne__, args)) == 3 and angular_range is None:
            angular_range = AngularRange.from_degrees(theta_min, theta_max, resolution)
        if isinstance(angular_range, AngularRange):
            return angular_range
        return None
    
    def _slice_angle(self, data, angular_range: AngularRange, keys=None) -> xr.Dataset:
        """Slice the dataset to the specified angular range."""
        if angular_range is None:
            msg = "angular_range cannot be None."
            raise ValueError(msg)
        theta_min = angular_range.min
        theta_max = angular_range.max
        datasets = {key: data[key] for key in keys} if keys else data
        for key, ds in datasets.items():
            theta = ds.theta.to_numpy()
            mask = (theta >= theta_min) & (theta <= theta_max)
            ds.sel(theta=theta[mask]).sortby('theta')
            datasets.update({key: ds})
        return  data
    
    def interpolate_angles(self, data: xr.Dataset, theta_min, theta_max, resolution, keys=None, suffix=None) -> dict[str, xr.Dataset]:
        """Interpolate the dataset to the specified angular range."""
        angular_range = self._set_angular_range(theta_min, theta_max, resolution, None)
        if angular_range is None:
            msg = "angular_range cannot be None."
            raise ValueError(msg)
        theta_min = angular_range.min
        theta_max = angular_range.max
        resolution = angular_range.resolution
        new_theta = angular_range.range.to_numpy()
        
        # Interpolate amplitude and phase
        datasets = {key: data[key] for key in keys} if keys else data
        result = {}
        for key, ds in datasets.items():
            ds_interp = ds.interp(theta=new_theta, method='linear')
            ds_interp['theta'] = new_theta
            ds_interp = ds_interp.assign_coords(theta=ds_interp.theta.astype(np.float64))
            ds_interp['amplitude'] = ds_interp['amplitude'].astype(np.float64)
            ds_interp['phase'] = ds_interp['phase'].astype(np.float64)
            _key = f"{key}_interp" if suffix is None else suffix
            ds_interp.attrs["name"] = _key
            result.update({_key: ds_interp})
        return result
        
    @staticmethod
    def _compute_fwhm(theta, amp):
        amp = np.asarray(amp, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)

        if len(amp) != len(theta):
            return np.nan

        max_val = np.max(amp)
        half_max = max_val - 3.

        # Find where it crosses the half max
        indices = np.where(amp >= half_max)[0]

        if len(indices) < 2:
            return np.nan

        left_idx = indices[0]
        right_idx = indices[-1]

        return np.abs(theta[right_idx] - theta[left_idx])

    def compute_fwhm(self, datasets, keys=None):
        """Compute the Full Width at Half Maximum (FWHM) for each frequency and polarization."""
        _datasets = {key: datasets[key] for key in keys} if keys else datasets
        result = {}
        for key, ds in _datasets.items():
            fwhm_da = xr.apply_ufunc(
            Beam._compute_fwhm,
            ds.theta.broadcast_like(ds.amplitude),
            ds.amplitude.astype(np.float64),
            input_core_dims=[["theta"], ["theta"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
            ds['fwhm'] = fwhm_da
            result.update({key: ds}) 
        return result

    def _select_main_lobe(self, data, keys=None):
        datasets = {key: data[key] for key in keys} if keys else data
        result = {}
        
        for _key, ds in datasets.items():
            # Verifica se há NaNs em fwhm
            fwhm_valid = not ds["fwhm"].isnull().any().compute()

            ds.attrs["main_lobe_valid"] = fwhm_valid

            if not fwhm_valid:
                result[_key] = ds  # mantém o dataset original, com atributo sinalizando
                continue

            theta = ds.theta
            fwhm_broadcasted = ds.fwhm.expand_dims(theta=theta)
            theta_broadcasted = theta.expand_dims({"polarization": ds.polarization, "freq": ds.freq})
            theta_broadcasted = theta_broadcasted.transpose("polarization", "freq", "theta")
            
            mask = np.abs(theta_broadcasted) <= fwhm_broadcasted
            masked_ds = ds.where(mask)
            masked_ds.attrs["main_lobe_valid"] = True
            result[masked_ds.attrs.get('name', _key)] = masked_ds
        
        return result


    def compute_gain(self, data, type="Antenna", keys=None):
        """Calculate the gain from the amplitude in dB."""
        datasets = {key: data[key] for key in keys} if keys else data
        result = {}

        # Substitui por versão mascarada apenas se tipo for MainLobe
        if type == "MainLobe":
            datasets = self._select_main_lobe(datasets, keys=keys)

        label_db = "directivity_db" if type == "Antenna" else "directivity_db_ML"
        label = "directivity" if type == "Antenna" else "directivity_ML"

        for _key, ds in datasets.items():
            if not ds.attrs.get("main_lobe_valid", True):  # se inválido, apenas salva e continua
                result[_key] = ds
                continue

            power = Beam._db_to_power(ds["amplitude"].astype("float64"))**2
            dtheta = ds.theta.diff(dim="theta")
            dtheta = dtheta.reindex(theta=ds.theta, method="nearest", tolerance=1e-8).fillna(0)

            # Max power across theta
            P_max = power.max(dim="theta")

            # Integral over theta
            P_integral = (power * dtheta).sum(dim="theta")

            # Directivity (linear scale)
            directivity = 2 * np.pi * P_max / P_integral
            directivity_db = 10 * np.log10(directivity)

            ds[label] = directivity
            ds[label_db] = directivity_db
            result[_key] = ds
        
        return result

    def simmetrize(self, data, keys=None):
        """Simmetrize the dataset by averaging over positive and negative angles."""
        datasets = {key: data[key] for key in keys} if keys else data
        result = {}
        for key, ds in datasets.items():
            theta = ds.theta.to_numpy()
            mask_pos = theta >= 0
            mask_neg = theta < 0
            
            # Average positive and negative angles
            pos_avg = ds.sel(theta=theta[mask_pos]).mean(dim='theta')
            neg_avg = ds.sel(theta=theta[mask_neg]).mean(dim='theta')
            
            # Combine results
            combined = xr.concat([pos_avg, neg_avg], dim='theta')
            combined.attrs['name'] = ds.attrs['name'] +  "_symmetrized"
            result.update({key: combined})
        return result
    
    @staticmethod
    def _db_to_power(db):
        return 10 ** (db / 10)    

    def compute_efficiency(self, data, keys=None):
        """Calculate the efficiency from the gain."""
        datasets = {key: data[key] for key in keys} if keys else data
        result = {}
        for key, ds in datasets.items():
            if ds.attrs.get("main_lobe_valid", True) is False:
                result[key] = ds
                continue
            # Check if both directivity and directivity_ML are present
            if "directivity" in ds and "directivity_ML" not in ds:
                msg = f"Dataset {key} does not contain 'directivity_ML' for efficiency calculation."
                raise ValueError(msg)
            if "directivity_ML" in ds and "directivity" not in ds:
                msg = f"Dataset {key} does not contain 'directivity' for efficiency calculation."
                raise ValueError(msg)
            if ("directivity" in ds) and ("directivity_ML" in ds):
                efficiency = ds.directivity_ML / ds.directivity
                ds["efficiency"] = efficiency
                result.update({ds.attrs['name']: ds})
            else:
                msg = f"Dataset {key} does not contain required fields for efficiency calculation."
                raise ValueError(msg)
            #result = data.update(result)
        return result
    
    def process(self, keys=None, suffix=None):
        """Process the dataset through all steps."""
        # Select the keys to process
        process_keys = keys if keys else list(self.datasets.keys())
        updated = self.datasets.copy()
        # Step 0: Simmetrize the datasets
        updated = self.simmetrize(self.datasets, keys=process_keys)
        # Step 1: Compute FWHM
        updated = self.compute_fwhm(self.datasets, keys=process_keys)

        # Step 2: Compute Gain
        updated = self.compute_gain(updated, keys=process_keys)
        
        updated = self.compute_gain(updated, keys=process_keys, type="MainLobe")

        # Step 3: Compute Efficiency
        updated = self.compute_efficiency(updated, keys=process_keys)

        # Merge with the untouched datasets
        untouched_keys = set(self.datasets.keys()) - set(process_keys)
        merged = {key: self.datasets[key] for key in untouched_keys}
        merged.update(updated)
        merged = {f"{key + "_" + suffix}": merged[key] for key, ds in merged.items()}
        # If a suffix is provided, rename the datasets
        # Update the main datasets
        self.datasets = merged
        
        return self.datasets
    
    
    def make_uv(self, keys=None, n_points=1000):
        """Create a UV dataset from the beam data."""
        datasets = {key: self.datasets[key] for key in keys} if keys else self.datasets
        return {key: self._create_uv_map(ds, key=key, n_points=n_points) for key, ds in datasets.items()}
            
    def _create_uv_map(self, ds, key=None, r=None, phi=None, n_points=1000):
        """Gera mapa UV interpolado a partir de um beam 1D."""
        theta_vals = ds.theta.astype(float).to_numpy()
        r = r if r is not None else theta_vals
        phi = phi if phi is not None else np.linspace(0, 2 * np.pi, n_points)

        def interpolate_2d(var):
            var_interp = var.interp(theta=r, method="linear")
            return var_interp.expand_dims(phi=phi, axis=-1)

        amp_2d = interpolate_2d(ds["amplitude"]).astype("float32")
        phase_2d = interpolate_2d(ds["phase"]).astype("float32")

        ds_uv = xr.Dataset(
            data_vars=dict(amplitude=amp_2d, phase=phase_2d),
            coords=dict(
                polarization=ds.polarization,
                freq=ds.freq,
                r=r,
                phi=phi,
            ),
            attrs=dict(name="uv_beam"),
        )
        r_rad = np.deg2rad(ds_uv.r.values)
        phi = ds_uv.phi.values

        u = np.sin(r_rad[:, None]) * np.cos(phi)
        v = np.sin(r_rad[:, None]) * np.sin(phi)

        result = ds_uv.assign_coords(u=(("r", "phi"), u), v=(("r", "phi"), v))
        return {"2_D": result}
    
    

#     def reconstruct_2d_uv(self):
#         # Simples reconstrução com simetria rotacional
#         u = np.linspace(-1, 1, 200)
#         v = np.linspace(-1, 1, 200)
#         U, V = np.meshgrid(u, v)
#         R = np.sqrt(U**2 + V**2)
#         result = []
#         for f in self.ds.freq.to_numpy():
#             for pol in self.ds.to_values:
#                 interp = interp1d(self.ds.theta, self.ds.sel(freq=f, pol=pol).amp, kind="cubic", bounds_error=False, fill_value=-100)
#                 amp2d = interp(np.degrees(np.arctan2(R, 1e-6)))
#                 result.append(xr.DataArray(amp2d, coords={"u": u, "v": v}, dims=["u", "v"]).expand_dims(freq=[f], pol=[pol]))
#         return xr.concat(result, dim="freq")

#     def view_2d_uv(self):
#         ds2d = self.reconstruct_2d_uv()
#         def plot(freq):
#             return hv.HoloMap({pol: hv.Image(ds2d.sel(freq=freq, pol=pol), kdims=["u", "v"]).opts(colorbar=True, title=pol) for pol in ds2d.pol.to_numpy()})
#         return pn.Row(pn.widgets.Select(options=list(ds2d.freq.values), name="Freq"), pn.bind(plot))

#     def reconstruct_spherical(self):
#         # Placeholder: requer projeção de beam para coordenadas esféricas
#         raise NotImplementedError("Reconstrução esférica ainda não implementada")

#     def create_airy_beam(self, fwhm, freq):
#         from scipy.special import j1
#         theta = np.linspace(-5, 5, 1000)
#         k = 2 * np.pi * freq * 1e6 / 3e8
#         a = np.radians(fwhm / 2.0)
#         x = k * np.sin(np.radians(theta)) * a
#         airy = (2 * j1(x) / x) ** 2
#         airy_db = 10 * np.log10(np.clip(airy, 1e-12, None))
#         return hv.Curve((theta, airy_db)).opts(title="Airy Beam", xlabel="Theta (°)", ylabel="Amp (dB)")

#     def profile_methods(self):
#         with Profiler() as prof, ResourceProfiler() as rprof:
#             self.compute_efficiencies()
#         visualize([prof, rprof])
