import numpy as np
import torch
import json
from scipy.interpolate import interp1d

class PerSampleInterpolator:
    """
    Interpolate a 1D spectrum from an existing domain to a fixed-size array.
    If do_normalize=True, min-max scale to [0,1].
    """

    def __init__(self, target_size=1000, method='linear', do_normalize=True):
        self.target_size = target_size
        self.method = method
        self.do_normalize = do_normalize

    def __call__(self, spectrum: np.ndarray, domain: np.ndarray) -> np.ndarray:
        """
        Interpolate 'spectrum' onto domain.min()..domain.max() with `target_size` points.
        If domain is invalid, returns zeros.
        """
        if spectrum is None or domain is None or len(spectrum) == 0 or len(domain) == 0:
            return np.zeros(self.target_size, dtype=np.float32)

        spectrum = np.asarray(spectrum, dtype=np.float32).ravel()
        domain   = np.asarray(domain,   dtype=np.float32).ravel()

        # Sort if needed
        if not np.all(np.diff(domain) > 0):
            idx = np.argsort(domain)
            domain   = domain[idx]
            spectrum = spectrum[idx]

        d_min, d_max = domain[0], domain[-1]
        if d_max <= d_min:
            return np.zeros(self.target_size, dtype=np.float32)

        # Create new domain
        new_domain = np.linspace(d_min, d_max, self.target_size, dtype=np.float32)

        method = self.method
        if method == 'cubic' and len(domain) < 4:
            method = 'linear'

        f = interp1d(domain, spectrum, kind=method, bounds_error=False, fill_value=0)
        resampled = f(new_domain)
        resampled = np.nan_to_num(resampled, nan=0.0).astype(np.float32)

        if self.do_normalize:
            mn, mx = resampled.min(), resampled.max()
            if mx > mn:
                resampled = (resampled - mn) / (mx - mn)

        return resampled


class SpectralPreprocessor:
    """
    If use_preloaded_domain = True, we load IR/H‑NMR/C‑NMR domain arrays from a JSON file
    (domain_file) during init, and use those for all samples. The dataset need only
    provide the raw spectral intensities (shape (B, L)).

    If use_preloaded_domain = False, we skip domain loading and behave as a pass-through
    or skip interpolation.
    """

    def __init__(
        self,
        use_preloaded_domain: bool = True,
        domain_file: str | None = "data_extraction\multimodal_spectroscopic_dataset\meta_data\spectrum_dimensions.json",
        # Interpolators for each modality
        ir_interpolator: PerSampleInterpolator = None,
        hnmr_interpolator: PerSampleInterpolator = None,
        cnmr_interpolator: PerSampleInterpolator = None
    ):
        self.use_preloaded_domain = use_preloaded_domain
        self.domain_file = domain_file
        self.ir_interpolator   = ir_interpolator
        self.hnmr_interpolator = hnmr_interpolator
        self.cnmr_interpolator = cnmr_interpolator

        # Always try to load domains
        self.ir_domain   = None
        self.h_nmr_domain= None
        self.c_nmr_domain= None
        if domain_file is not None:
            try:
                with open(self.domain_file, "r") as f:
                    sp_dims = json.load(f)
                # Grab arrays from JSON
                self.ir_domain    = np.array(sp_dims["ir_spectra"]["dimensions"],   dtype=np.float32)
                self.h_nmr_domain = np.array(sp_dims["h_nmr_spectra"]["dimensions"],dtype=np.float32)
                self.c_nmr_domain = np.array(sp_dims["c_nmr_spectra"]["dimensions"],dtype=np.float32)
                print(f"[SpectralPreprocessor] Loaded IR domain of shape {self.ir_domain.shape}")
                print(f"[SpectralPreprocessor] Loaded H-NMR domain of shape {self.h_nmr_domain.shape}")
                print(f"[SpectralPreprocessor] Loaded C-NMR domain of shape {self.c_nmr_domain.shape}")
            except Exception as e:
                print(f"[SpectralPreprocessor] Warning: Failed to load domains from {domain_file}: {e}")

    def _process_modality(self, batch_data: torch.Tensor, domain_array: np.ndarray,
                          interpolator: PerSampleInterpolator):
        """
        batch_data: shape (B, L) or None
        domain_array: 1D array with shape (N,) from preloaded domains
        interpolator: PerSampleInterpolator
        """
        if batch_data is None or interpolator is None:
            return None

        # Always interpolate using the preloaded domain
        batch_data_np = batch_data.cpu().numpy()  # (B, L)
        B, L = batch_data_np.shape

        results = []
        for i in range(B):
            spectrum_i = batch_data_np[i]  # shape (L,)
            res_i = interpolator(spectrum_i, domain_array)  # shape (target_size,)
            results.append(res_i)

        stacked = np.stack(results, axis=0)  # (B, target_size)
        # Convert to tensor and move to same device as input
        tensor = torch.from_numpy(stacked).unsqueeze(1)  # (B,1,target_size)
        return tensor.to(device=batch_data.device, dtype=batch_data.dtype)

    def __call__(self, ir_data, h_nmr_data, c_nmr_data):
        """
        Each input is shape (B, L) or None.
        Returns up to 3 Tensors => (B,1,R) each, or None if skipping or no data.
        """
        x_ir = self._process_modality(ir_data, self.ir_domain,   self.ir_interpolator)
        x_hnmr = self._process_modality(h_nmr_data, self.h_nmr_domain, self.hnmr_interpolator)
        x_cnmr = self._process_modality(c_nmr_data, self.c_nmr_domain, self.cnmr_interpolator)
        return x_ir, x_hnmr, x_cnmr
