from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CalibrationPoint:
    pixel_x: float
    pixel_y: float
    x: float
    y: float

def _region(region):
    r = np.asarray(region, dtype=float)
    if r.shape != (4,) or not np.isfinite(r).all() or r[0] >= r[2] or r[1] >= r[3]:
        raise ValueError('invalid region')
    return r

def _inside(x, y, r):
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]

class XYCalibration:
    def __init__(self, coefficients, pixel_region, physical_region):
        self.coefficients = coefficients
        self.pixel_region = _region(pixel_region)
        self.physical_region = _region(physical_region)

    @classmethod
    def fit(cls, points, *, pixel_region, physical_region):
        values = np.asarray([(p.pixel_x,p.pixel_y,p.x,p.y) for p in points], dtype=float)
        if values.ndim != 2 or values.shape[0] < 6 or values.shape[1] != 4 or not np.isfinite(values).all():
            raise ValueError('at least six finite points required')
        if len(np.unique(values[:, :2], axis=0)) != len(values):
            raise ValueError('duplicate pixel points')
        x,y = values[:,0],values[:,1]
        design = np.column_stack((np.ones(len(x)),x,y,x*x,x*y,y*y))
        coefficients, _, rank, _ = np.linalg.lstsq(design, values[:,2:], rcond=None)
        if rank != 6: raise ValueError('calibration points are rank deficient')
        return cls(coefficients, pixel_region, physical_region)

    def map_pixel(self, x, y):
        x,y = float(x),float(y)
        if not np.isfinite([x,y]).all() or not _inside(x,y,self.pixel_region):
            raise ValueError('pixel outside safe region')
        result = np.array([1,x,y,x*x,x*y,y*y]) @ self.coefficients
        self.validate_physical(*result)
        return tuple(float(v) for v in result)

    def validate_physical(self, x, y):
        if not np.isfinite([x,y]).all() or not _inside(x,y,self.physical_region):
            raise ValueError('position outside physical region')
