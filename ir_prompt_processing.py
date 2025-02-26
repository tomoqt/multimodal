import numpy as np
from scipy.interpolate import interp1d

def process_ir(ir: np.ndarray, interpolation_points: int = 400) -> str:
    """
    Process an IR spectrum by reinterpolating over a fixed number of points,
    normalizing intensities to range [0,100] and rounding them to integer strings.
    Returns a string starting with an 'IR' prefix followed by space-separated token values.
    """
    original_x = np.linspace(400, 4000, 1800)
    interpolation_x = np.linspace(400, 4000, interpolation_points)
    interp = interp1d(original_x, ir)
    interp_ir = interp(interpolation_x)
    
    # Normalize
    interp_ir = interp_ir + abs(min(interp_ir))
    interp_ir = (interp_ir / max(interp_ir)) * 100
    interp_ir = np.round(interp_ir, decimals=0).astype(int).astype(str)
    
    return 'IR ' + ' '.join(interp_ir) + ' ' 