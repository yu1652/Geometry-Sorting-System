"""Plan a straight negative-X push from observed footprints, without motion I/O.

The 25 mm plate may contact a part off center. Travel limits are never enlarged.
Both the plate's path and the part's swept footprint must remain clear. Existing
contact may be left behind, but may not grow during the push. This is a planar
clearance check, not a simulation of friction, tipping, or stacked objects.
"""
import math
import cv2
import numpy as np
from motion_limits import X_MIN_MM, Y_MIN_MM, X_MAX_MM, Y_MAX_MM


def _hull(points):
    points=np.asarray(points,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError('invalid object footprint')
    hull=cv2.convexHull(points).reshape(-1,2)
    if len(hull)<3 or cv2.contourArea(hull)<=0:
        raise ValueError('degenerate object footprint')
    return hull


def _rectangle(x1,y1,x2,y2):
    return np.array(((x1,y1),(x2,y1),(x2,y2),(x1,y2)),np.float32)


def _area(a,b):
    return float(cv2.intersectConvexConvex(a,b)[0])


def plan_push_contact(footprint,obstacles,preferred,region,*,width_mm,gap_mm,
                      side_clearance_mm=.5,sweep_tolerance_mm2=.25):
    """Return (start_x, y) or None.

    ``side_clearance_mm`` is the extra Y room demanded beside the plate and
    ``sweep_tolerance_mm2`` how much extra overlap the object's travel may pick
    up. Both are tolerances for segmentation noise on touching parts, not
    permission to drive through a neighbour: default values keep the strict
    historical behaviour, and the caller sets them per release profile.
    """
    part=_hull(footprint)
    obstacles=[_hull(poly) for poly in obstacles]
    values=np.asarray(list(preferred)+list(region)+[width_mm,gap_mm,side_clearance_mm,
                                                    sweep_tolerance_mm2],dtype=float)
    if (values.shape!=(10,) or not np.isfinite(values).all() or width_mm<=0 or gap_mm<0
            or not 0<=side_clearance_mm<=5 or not 0<=sweep_tolerance_mm2<=500):
        raise ValueError('invalid pusher contact settings')
    xmin,ymin,xmax,ymax=map(float,region)
    if xmin>=xmax or ymin>=ymax:
        raise ValueError('invalid contact travel region')
    xmin,ymin,xmax,ymax=max(X_MIN_MM,xmin),max(Y_MIN_MM,ymin),min(X_MAX_MM,xmax),min(Y_MAX_MM,ymax)
    rear=float(part[:,0].max())
    start_x=max(float(preferred[0]),rear+float(gap_mm))
    if not xmin<=start_x<=xmax or float('%.3f'%start_x)<=10 or ymin>ymax:
        return None
    # A lower or narrower contact cannot make an object pass through a neighbour.
    end=part-np.array((max(0,rear-10),0),np.float32)
    swept=_hull(np.concatenate((part,end)))
    for other in obstacles:
        if _area(swept,other)>_area(part,other)+sweep_tolerance_mm2:
            return None
    low,high=float(part[:,1].min()),float(part[:,1].max())
    half=float(width_mm)/2
    minimum_overlap=min(half,(high-low)/2)
    # Include exact preferred/boundary values, then search at 0.5 mm spacing.
    desired=float(preferred[1])
    choices=[desired,min(ymax,max(ymin,desired)),ymin,ymax]
    choices.extend(np.arange(math.ceil(ymin*2)/2,ymax+.001,.5).tolist())
    for y in sorted(set(choices),key=lambda value:(abs(value-desired),value)):
        if not ymin<=y<=ymax:
            continue
        if min(high,y+half)-max(low,y-half)<minimum_overlap:
            continue
        # A triangular/tilted outline can overlap the Y band only beyond the
        # stop line. Require real footprint contact during the commanded stroke.
        contact_area,contact=cv2.intersectConvexConvex(part,
            _rectangle(10,y-half,start_x+.5,y+half))
        if contact is None or contact_area<=.001 or float(contact[:,:,0].max())<=10.001:
            continue
        # The measured X thickness is intentionally not changed; retain the
        # configured rear gap and the profile's clearance around the plate path.
        path=_rectangle(10,y-half-side_clearance_mm,start_x+.5,y+half+side_clearance_mm)
        if any(_area(path,other)>.001 for other in obstacles):
            continue
        return float(start_x),float(y)
    return None
