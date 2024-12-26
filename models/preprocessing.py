import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import Optional, Tuple, List, Union
import logging

class GlobalWindowResampler:
    """Resample spectrum to global window with fixed size"""
    
    def __init__(self, target_size: int, window: List[float] = [200, 4000], method: str = 'cubic'):
        self.target_size = target_size
        self.window = self.validate_window(window)
        self.method = method
        self.global_domain = np.linspace(window[0], window[1], target_size)
        
    def validate_window(self, window: List[float]) -> List[float]:
        if not isinstance(window, (list, tuple, np.ndarray)) or len(window) != 2:
            raise ValueError("Window must be a list/tuple of [min, max] wavenumbers")
        if window[0] >= window[1]:
            raise ValueError(f"Invalid window range: [{window[0]}, {window[1]}]")
        return [float(window[0]), float(window[1])]
    
    def __call__(self, spectrum: np.ndarray, domain: Union[np.ndarray, List[float]]) -> np.ndarray:
        if isinstance(domain, list):
            domain = np.array(domain)
        if isinstance(spectrum, list):
            spectrum = np.array(spectrum)
            
        # Sort if needed
        if not np.all(np.diff(domain) > 0):
            sort_idx = np.argsort(domain)
            domain = domain[sort_idx]
            spectrum = spectrum[sort_idx]
            
        # Create padded spectrum
        padded_spectrum = np.zeros(self.target_size)
        
        # Determine window bounds
        min_wave = max(np.min(domain), self.window[0])
        max_wave = min(np.max(domain), self.window[1])
        
        # Find indices
        start_idx = np.searchsorted(self.global_domain, min_wave)
        end_idx = np.searchsorted(self.global_domain, max_wave, side='right')
        
        if start_idx < end_idx:
            section_domain = self.global_domain[start_idx:end_idx]
            mask = (domain >= min_wave) & (domain <= max_wave)
            window_domain = domain[mask]
            window_spectrum = spectrum[mask]
            
            if len(window_domain) > 0:
                if len(window_domain) > 3 and self.method == 'cubic':
                    try:
                        # Add small epsilon to avoid division by zero
                        eps = 1e-10
                        if np.any(np.diff(window_domain) < eps):
                            raise ValueError("Domain points too close together")
                        interpolator = CubicSpline(window_domain, window_spectrum, extrapolate=False)
                    except ValueError:
                        interpolator = interp1d(window_domain, window_spectrum, kind='linear', 
                                             bounds_error=False, fill_value=0)
                else:
                    interpolator = interp1d(window_domain, window_spectrum, kind='linear', 
                                         bounds_error=False, fill_value=0)
                    
                interpolated_values = interpolator(section_domain)
                interpolated_values = np.nan_to_num(interpolated_values, 0)
                padded_spectrum[start_idx:end_idx] = interpolated_values
                
        return padded_spectrum

class GlobalWindowResampler2D:
    """Resample 2D spectrum (like HSQC) to global window with fixed size"""
    
    def __init__(self, target_size: Tuple[int, int], window_h: List[float] = [0, 12], window_c: List[float] = [0, 200], method: str = 'linear'):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
            
        self.target_size = target_size
        self.window_h = self.validate_window(window_h, "H")
        self.window_c = self.validate_window(window_c, "C")
        self.method = method
        
        # Create global domains for both dimensions
        self.global_domain_h = np.linspace(window_h[0], window_h[1], target_size[0])
        self.global_domain_c = np.linspace(window_c[0], window_c[1], target_size[1])
        
    def validate_window(self, window: List[float], dim_name: str) -> List[float]:
        if not isinstance(window, (list, tuple, np.ndarray)) or len(window) != 2:
            raise ValueError(f"{dim_name} window must be a list/tuple of [min, max]")
        if window[0] >= window[1]:
            raise ValueError(f"Invalid {dim_name} window range: [{window[0]}, {window[1]}]")
        return [float(window[0]), float(window[1])]
    
    def __call__(self, spectrum: np.ndarray, domain_h: Union[np.ndarray, List[float]], domain_c: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Resample 2D spectrum to target size.
        
        Args:
            spectrum: 2D array of intensity values
            domain_h: 1D array of proton chemical shifts
            domain_c: 1D array of carbon chemical shifts
            
        Returns:
            Resampled 2D spectrum of shape target_size
        """
        if isinstance(domain_h, list):
            domain_h = np.array(domain_h)
        if isinstance(domain_c, list):
            domain_c = np.array(domain_c)
        if isinstance(spectrum, list):
            spectrum = np.array(spectrum)
            
        # Ensure spectrum is 2D
        if spectrum.ndim != 2:
            raise ValueError(f"Expected 2D spectrum, got shape {spectrum.shape}")
            
        # Create empty output array
        resampled = np.zeros(self.target_size)
        
        # Determine window bounds
        min_h = max(np.min(domain_h), self.window_h[0])
        max_h = min(np.max(domain_h), self.window_h[1])
        min_c = max(np.min(domain_c), self.window_c[0])
        max_c = min(np.max(domain_c), self.window_c[1])
        
        # Find indices in global domain
        h_start = np.searchsorted(self.global_domain_h, min_h)
        h_end = np.searchsorted(self.global_domain_h, max_h, side='right')
        c_start = np.searchsorted(self.global_domain_c, min_c)
        c_end = np.searchsorted(self.global_domain_c, max_c, side='right')
        
        if h_start < h_end and c_start < c_end:
            # Get the section of the global domain we'll interpolate to
            h_points = self.global_domain_h[h_start:h_end]
            c_points = self.global_domain_c[c_start:c_end]
            
            # Create meshgrid for interpolation
            H, C = np.meshgrid(domain_h, domain_c, indexing='ij')
            Hnew, Cnew = np.meshgrid(h_points, c_points, indexing='ij')
            
            try:
                # Try using scipy's griddata for interpolation
                from scipy.interpolate import griddata
                points = np.column_stack((H.flatten(), C.flatten()))
                values = spectrum.flatten()
                
                # Remove any NaN or infinite values
                valid_mask = np.isfinite(values)
                if np.any(valid_mask):
                    interpolated = griddata(
                        points[valid_mask],
                        values[valid_mask],
                        (Hnew, Cnew),
                        method=self.method,
                        fill_value=0
                    )
                    
                    # Fill any NaN values with 0
                    interpolated = np.nan_to_num(interpolated, 0)
                    
                    # Place in output array
                    resampled[h_start:h_end, c_start:c_end] = interpolated
                    
            except Exception as e:
                logging.warning(f"Interpolation failed: {str(e)}. Returning zero array.")
                
        return resampled

class Normalizer:
    """Normalize spectrum to [0,1] range"""
    
    def __call__(self, spectrum: Union[np.ndarray, List[float]], domain: Optional[np.ndarray] = None) -> np.ndarray:
        if isinstance(spectrum, list):
            spectrum = np.array(spectrum)
            
        non_zero_mask = spectrum != 0
        if non_zero_mask.any():
            min_val = np.min(spectrum[non_zero_mask])
            max_val = np.max(spectrum[non_zero_mask])
            
            if not np.isclose(min_val, max_val):
                spectrum = spectrum.copy()
                spectrum[non_zero_mask] = (spectrum[non_zero_mask] - min_val) / (max_val - min_val)
                
        return spectrum 