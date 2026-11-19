"""Tray foreground components relative to one operator-confirmed empty image.

No camera, serial, model or motion access. ``detect`` returns ROI-local xyxy and
an acceptance flag in [0,1], rather than a learned probability. Different HSV
regions keep touching colored parts separate. Same-color components can be
split when model detections and visible edges agree. Geometry and motion
safety remain the caller's responsibility.
Compact sharp neutral intensity changes are retained, while soft neutral
changes can block as shadow ambiguities. A hard shadow identical to a black
object cannot be distinguished from that object using one RGB frame alone.
"""
import hashlib
import cv2
import numpy as np

GROUPS = (('warm',0,38),('green',38,88),('cyan_gray',88,112),('blue',112,140),
          ('purple',140,170),('red_wrap',170,180))
COLORED_SPLIT_GROUPS = ('warm', 'green', 'cyan_color', 'blue', 'purple', 'red_wrap')
SPLIT_GROUPS = COLORED_SPLIT_GROUPS + ('neutral_bright', 'neutral_dark', 'dark_part')
SPLIT_REASON = 'model_and_visible_color_boundary'


def _validate_image(image, label):
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
            or image.shape[2]!=3 or min(image.shape[:2])<64):
        raise ValueError(label+' must be a uint8 HxWx3 BGR image of at least 64x64')


def _tray_split_hints(hints, width, height):
    for hint in hints:
        try:
            x1, y1, x2, y2, score = map(float, hint)
        except (TypeError, ValueError):
            continue
        if (.60 <= score <= 1 and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                and min(x2 - x1, y2 - y1) >= 16):
            yield (x1, y1, x2, y2), score


def _hint_support_masks(region, x, y, hints, minimum_area):
    height, width = region.shape
    yy, xx = np.ogrid[y:y + height, x:x + width]
    for x1, y1, x2, y2 in hints:
        tight = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
        if np.count_nonzero(region & tight) >= max(minimum_area, .25 * (x2 - x1) * (y2 - y1)):
            yield (xx >= x1 - 6) & (xx < x2 + 6) & (yy >= y1 - 6) & (yy < y2 + 6)


def _split_supported_component(image, region, x, y, hints, minimum_area, seam_contrast):
    """Partition only observed pixels, using separate model seeds and visible seams."""
    strong_hints, secondary_hints = hints
    supports = list(_hint_support_masks(region, x, y, strong_hints, minimum_area))
    if len(supports) < 2:
        return ()
    coverage = np.sum(supports, axis=0)
    remaining = region & (coverage == 0)
    allowed_uncovered = .02 * np.count_nonzero(region)
    if np.count_nonzero(remaining) > allowed_uncovered:
        # A single lower-confidence detection may explain a missing part of
        # the observed region. The two strong seeds and the seams still apply.
        completing = [mask for mask in _hint_support_masks(region, x, y, secondary_hints, minimum_area)
                      if np.count_nonzero(remaining & mask) >= minimum_area
                      and np.count_nonzero(remaining & ~mask) <= allowed_uncovered]
        if len(completing) != 1:
            return ()
        supports.append(completing[0])
        coverage += completing[0]

    markers = np.ones(region.shape, dtype=np.int32)
    markers[region] = 0
    costs = []
    for index, support in enumerate(supports):
        core = region & support & (coverage == 1)
        # The exclusive seed excludes the box overlap. The full part still
        # has to satisfy the original minimum area and floor-support checks.
        if np.count_nonzero(core) < min(minimum_area, 600):
            return ()
        markers[core] = index + 2
        distance = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        costs.append(distance + 1000 * (~support))
    labels = cv2.watershed(image.copy(), markers) - 2
    nearest = np.argmin(costs, axis=0)
    labels[labels < 0] = nearest[labels < 0]
    for index, support in enumerate(supports):
        outside = (labels == index) & ~support
        labels[outside] = nearest[outside]
    parts = [region & (labels == index) for index in range(len(supports))]

    kernel = np.ones((5, 5), dtype=np.uint8)
    interior = cv2.erode(region.astype(np.uint8), kernel) > 0
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel).max(axis=2)
    for part in parts:
        if np.count_nonzero(part) < minimum_area:
            return ()
        # Sample the internal cut, away from the foreground/background edge.
        neighbors = cv2.dilate(part.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
        seam = gradient[interior & neighbors & ~part]
        if len(seam) < 8 or np.median(seam) < seam_contrast:
            return ()
        count, _, stats, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8), connectivity=8)
        if count != 2 or min(stats[1, 2:4]) < 16 or stats[1, cv2.CC_STAT_AREA] > 23000:
            return ()
    return parts


def _has_neutral_split_outline(image, part, region, floor):
    mask = part.astype(np.uint8)
    _, _, width, height = cv2.boundingRect(mask)
    if np.count_nonzero(part) < .4 * width * height:
        return False
    kernel = np.ones((5, 5), dtype=np.uint8)
    inner = part & (cv2.erode(mask, kernel) == 0)
    outer = (cv2.dilate(mask, kernel) > 0) & ~region & floor
    if not inner.any() or not outer.any():
        return False
    return np.linalg.norm(np.median(image[inner], axis=0) - np.median(image[outer], axis=0)) >= 30


def _servo_cable_support(hsv):
    """Find narrow red leads entering the image, including their nearby strands.

    Color alone never excludes a component. A lead must reach an image edge,
    stay narrow along its entire observed length and have a long thin body.
    This support is only evidence for rejecting whole thin components below;
    it is not a cutout that erases pixels from a real part beside/under a lead.
    """
    red = (((hsv[:,:,0] <= 12) | (hsv[:,:,0] >= 165))
           & (hsv[:,:,1] >= 70) & (hsv[:,:,2] >= 35)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    cable = np.zeros_like(red)
    regions = []
    height, width = red.shape
    for label in range(1, count):
        x,y,w,h,area = map(int, stats[label])
        if area < 250 or not (x <= 2 or y <= 2 or x+w >= width-2 or y+h >= height-2):
            continue
        component = (labels == label).astype(np.uint8)
        radius = float(cv2.distanceTransform(component, cv2.DIST_L2, 5).max())
        length = area / max(1., 2*radius)
        if radius > 12 or length < 80 or length / max(1., 2*radius) < 4:
            continue
        cable |= component
        regions.append({'box_roi':[x,y,x+w,y+h], 'max_radius_px':radius,
                        'length_lower_bound_px':length})
    return cable > 0, cv2.dilate(cable,np.ones((35,35),np.uint8)) > 0, regions


class TrayForegroundDetector:
    def __init__(self, reference_roi, pale_hints=()):
        _validate_image(reference_roi, 'reference')
        self.ref = reference_roi.copy()
        self.pale_hints = pale_hints
        self._reference_sha256 = hashlib.sha256(self.ref.tobytes()).hexdigest()
        self.last_diagnostics = {}
        self.ref_hsv = cv2.cvtColor(self.ref, cv2.COLOR_BGR2HSV)
        self.ref_float = self.ref.astype(np.float32)
        self.ref_chroma = self.ref_float / np.maximum(self.ref_float.sum(2, keepdims=True), 1)
        self.reference_floor = ((self.ref_hsv[:, :, 1] < 45) & (self.ref_hsv[:, :, 2] > 135))
        floor = self.reference_floor.astype(np.uint8)*255
        floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        contours = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not contours:
            raise ValueError('Reference has no neutral bright tray surface')
        self.work = np.zeros(floor.shape, np.uint8)
        cv2.drawContours(self.work, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        if np.count_nonzero(self.work) < 5000:
            raise ValueError('Reference must contain at least 5000 valid tray-surface pixels')
        self.completion_work = np.zeros(floor.shape,np.uint8)
        hull = cv2.convexHull(max(contours,key=cv2.contourArea))
        cv2.drawContours(self.completion_work,[hull],-1,255,cv2.FILLED)
        # Allow an observed, connected part to extend 32 pixels beyond the
        # reference floor. Seeds still require actual floor replacement;
        # detached objects outside the floor cannot become new targets.
        self.completion_work = cv2.dilate(self.completion_work,np.ones((65,65),np.uint8))

    def _add_model_splits(self, group, image, hsv, delta_v, cleaned, labels, stats,
                          cable_labels, hints, minimum_area, detections, merged_mask):
        if group not in COLORED_SPLIT_GROUPS:
            hints = (hints[0], ())
        height, width = labels.shape
        for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if area < 2 * minimum_area or label in cable_labels or x + w >= width - 1:
                continue
            x1, y1 = max(0, x - 3), max(0, y - 3)
            x2, y2 = min(width, x + w + 3), min(height, y + h + 3)
            crop = np.s_[y1:y2, x1:x2]
            region = labels[crop] == label
            if np.any(region & (self.completion_work[crop] == 0)):
                continue
            local_image = image[crop]
            seam_contrast = 8 if group == 'dark_part' else 15
            parts = _split_supported_component(local_image, region, x1, y1, hints, minimum_area,
                                               seam_contrast)
            floor = self.reference_floor[crop] & (self.work[crop] > 0)
            if group == 'dark_part':
                floor &= (hsv[crop][:, :, 2] < 32) & (delta_v[crop] < -80)
            if not parts or any(np.count_nonzero(part & floor) < minimum_area for part in parts):
                continue
            if group.startswith('neutral_') and any(not _has_neutral_split_outline(local_image, part, region, floor)
                                                    for part in parts):
                continue
            for part in parts:
                ys, xs = np.nonzero(part)
                box = [max(0, int(xs.min()) + x1 - 2), max(0, int(ys.min()) + y1 - 2),
                       min(width, int(xs.max()) + x1 + 3), min(height, int(ys.max()) + y1 + 3)]
                observed = np.zeros(labels.shape, dtype=bool)
                observed[crop] = part
                detections.append({'box_roi': box, 'core_box_roi': box.copy(), 'group': group,
                                   'area': int(len(xs)), 'acceptance_score': 1.0,
                                   'reason': SPLIT_REASON,
                                   'median_saturation': float(np.median(hsv[:, :, 1][observed])),
                                   'median_value_difference': float(np.median(delta_v[observed])),
                                   '_component_mask': observed.astype(np.uint8) * 255,
                                   '_observed_support_mask': observed,
                                   '_dark_support_mask': observed if group == 'dark_part' else None})
            cleaned[crop][region] = 0
            labels[crop][region] = 0
            merged_mask[crop][region] = 255

    def _detect_diagnostics(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cable_core, cable_support, cable_regions = _servo_cable_support(hsv)
        support = ((self.work > 0) & (hsv[:, :, 1] < 25) & (self.ref_hsv[:, :, 1] < 25)
                   & (hsv[:, :, 2] > 80) & (self.ref_hsv[:, :, 2] > 80))
        if np.count_nonzero(support) < 5000:
            return [], {'reason': 'insufficient_stable_background', 'blocking_regions': [],
                        'hold_reason': '底板参考无法稳定匹配，请保持取景或重新保存底板'}, np.zeros(frame.shape[:2], np.uint8)
        ratio = frame.astype(np.float32) / np.maximum(self.ref_float, 1)
        gain = np.clip(np.median(ratio[support], axis=0), .6, 1.6)
        normalized = np.clip(frame.astype(np.float32)/gain, 0, 255)
        quantized = np.rint(normalized).astype(np.uint8)
        nhsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
        chroma = normalized / np.maximum(normalized.sum(2, keepdims=True), 1)
        distance = np.sqrt(np.square(chroma-self.ref_chroma).sum(2))
        delta_v = nhsv[:, :, 2].astype(float)-self.ref_hsv[:, :, 2].astype(float)
        colored = ((distance > .027) & (nhsv[:, :, 1] > 30) & (nhsv[:, :, 2] > 40))
        neutral_bright = ((delta_v > 22) & (nhsv[:, :, 1] < 45))
        neutral_dark = ((delta_v < -22) & (nhsv[:, :, 1] < 45))
        # Hue is unstable near black: one plastic part can straddle several
        # color bands. A strong very-dark floor replacement seeds one real
        # connected low-value body without changing normal color thresholds.
        dark_body = (nhsv[:,:,2] < 50) & (delta_v < -50)
        dark_core = (nhsv[:,:,2] < 32) & (delta_v < -80) & self.reference_floor
        masks = [(name, colored & (nhsv[:, :, 0] >= lower) & (nhsv[:, :, 0] < upper)
                  & (nhsv[:, :, 1] > (70 if name in ('warm', 'red_wrap') else 60 if name == 'green' else 45 if name == 'cyan_gray' else 30)))
                 for name, lower, upper in GROUPS]
        # Gray foam and saturated cyan plastic share a hue. Partition their
        # chroma before connectivity, so touching edges cannot create a single
        # oversized body or let one part claim the other's rear face.
        cyan=next(mask for name,mask in masks if name=='cyan_gray')
        masks=[(name,mask & (nhsv[:,:,1]<120) if name=='cyan_gray' else mask)
               for name,mask in masks]
        masks.append(('cyan_color',cyan & (nhsv[:,:,1]>=120)))
        masks = [('dark_part',dark_body)] + masks
        masks.extend((('neutral_bright', neutral_bright),('neutral_dark', neutral_dark)))
        detections, rejected, blocking = [], [], []
        merged_mask = np.zeros(frame.shape[:2], np.uint8)
        split_hints = tuple(_tray_split_hints(self.pale_hints, frame.shape[1], frame.shape[0]))
        strong_hints = tuple(box for box, score in split_hints if score >= .72)
        secondary_hints = tuple(box for box, score in split_hints if score < .72)
        for group, unmasked in masks:
            if group=='cyan_gray':
                # Recover pale patches of an already substantial, well-lit
                # gray face. Low-saturation rails and dark-body fragments do
                # not seed this completion.
                relaxed=unmasked | (colored & (nhsv[:,:,0]>=88) & (nhsv[:,:,0]<112)
                                     & (nhsv[:,:,1]>35) & (nhsv[:,:,1]<120) & (nhsv[:,:,2]>100))
                _,relaxed_labels,_,_=cv2.connectedComponentsWithStats(relaxed.astype(np.uint8),connectivity=8)
                for label in np.unique(relaxed_labels[unmasked & (self.work>0)]):
                    if label==0:
                        continue
                    seed=(relaxed_labels==label)&unmasked&(self.work>0)
                    if np.count_nonzero(seed)>=500 and np.median(nhsv[:,:,2][seed])>100:
                        unmasked=unmasked|(relaxed_labels==label)
            minimum_area = 1200 if group.startswith('neutral_') or group=='dark_part' else 500
            mask = unmasked & (self.work > 0)
            cleaned = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # Completion does not create seeds. Only original-work components
            # that pass the existing area/shape gates can claim these labels.
            raw_clean = cv2.morphologyEx(unmasked.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            raw_clean = cv2.morphologyEx(raw_clean,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            _,raw_labels,raw_stats,_ = cv2.connectedComponentsWithStats(raw_clean,connectivity=8)
            cable_labels = set()
            for label in np.unique(raw_labels[cable_support]):
                if label == 0:
                    continue
                body = raw_labels == label
                if group in ('warm','red_wrap'):
                    # Red fragments must belong to the actual edge-connected
                    # red lead, not merely be near it after support dilation.
                    if np.count_nonzero(body & cable_core) < .8*np.count_nonzero(body):
                        continue
                elif group in ('dark_part','neutral_dark','neutral_bright'):
                    # A dark companion must independently reach the frame
                    # edge and be a curved strand or a rejected thin sliver.
                    # Preserve every compact solid, including a small one.
                    ys,xs=np.where(body)
                    if not (xs.min()<=2 or ys.min()<=2 or xs.max()>=frame.shape[1]-3
                            or ys.max()>=frame.shape[0]-3):
                        continue
                    bw,bh=int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)
                    hull=cv2.convexHull(cv2.findNonZero(body.astype(np.uint8)))
                    solidity=np.count_nonzero(body)/max(1.,float(cv2.contourArea(hull)))
                    if solidity>=.6 and max(bw,bh)/min(bw,bh)<=4.5:
                        continue
                else:
                    # Colored neighbours have no independent cable identity.
                    continue
                # Whole-body overlap plus a thickness check protects a solid
                # red part and any larger part touched by an attached lead.
                if np.count_nonzero(body & cable_support) < .95*np.count_nonzero(body):
                    continue
                radius = float(cv2.distanceTransform(body.astype(np.uint8),cv2.DIST_L2,5).max())
                length = np.count_nonzero(body) / max(1.,2*radius)
                # Proximity alone cannot identify an adjacent small part as
                # another strand. Require its own long thin body as well.
                if radius <= 12 and length >= 60 and length/max(1.,2*radius) >= 4:
                    cable_labels.add(int(label))
            if group in SPLIT_GROUPS and len(strong_hints) >= 2:
                self._add_model_splits(group, quantized, nhsv, delta_v, cleaned, raw_labels,
                                       raw_stats, cable_labels, (strong_hints, secondary_hints), minimum_area, detections,
                                       merged_mask)
            contours = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            raw_owners = {}
            for contour in contours:
                area = float(cv2.contourArea(contour))
                x, y, w, h = cv2.boundingRect(contour)
                if cable_labels:
                    seed = np.zeros(frame.shape[:2],np.uint8)
                    cv2.drawContours(seed,[contour],-1,1,cv2.FILLED)
                    ids = set(int(v) for v in np.unique(raw_labels[seed > 0]) if v)
                    if ids and ids.issubset(cable_labels):
                        rejected.append({'box_roi':[x,y,x+w,y+h], 'group':group,
                                         'area':area, 'reason':'servo_cable'})
                        continue
                # A reference difference is not itself an object boundary.
                # Low-chroma shadows, including blue-tinted floor revealed by
                # a moved cable, must have a visible edge in the current frame
                # before raw connectivity can promote them to an obstacle.
                if area>=minimum_area and group!='dark_part':
                    seed=np.zeros(frame.shape[:2],np.uint8)
                    cv2.drawContours(seed,[contour],-1,1,cv2.FILLED)
                    seed_pixels=seed>0
                    if np.median(nhsv[:,:,1][seed_pixels])<70:
                        inner_seed=cv2.erode(seed,np.ones((5,5),np.uint8))>0
                        band_seed=seed_pixels&~inner_seed
                        outer_seed=cv2.dilate(seed,np.ones((5,5),np.uint8))>0
                        outer_seed &= ~seed_pixels & (self.work>0)
                        if band_seed.any() and outer_seed.any():
                            contrast_v=abs(float(np.median(nhsv[:,:,2][band_seed]))-float(np.median(nhsv[:,:,2][outer_seed])))
                            contrast_rgb=float(np.linalg.norm(np.median(normalized[band_seed],axis=0)-np.median(normalized[outer_seed],axis=0)))
                            ids=np.unique(raw_labels[seed_pixels]);ids=ids[ids!=0]
                            observed_seed=np.isin(raw_labels,ids).astype(np.uint8)
                            points=cv2.findNonZero(observed_seed)
                            solidity=(np.count_nonzero(observed_seed)/max(1.,float(cv2.contourArea(cv2.convexHull(points))))
                                      if points is not None else 0.)
                            # Use the completed component, since a cable in
                            # the reference can cut a real compact object into
                            # irregular seeds. Thin diffuse changes stay weak
                            # after completion; solid pale parts do not.
                            if solidity<.85 and contrast_v<18 and contrast_rgb<30:
                                rejected.append({'box_roi':[x,y,x+w,y+h],'group':group,'area':area,
                                                 'reason':'no_current_object_boundary',
                                                 'current_boundary':{'value_contrast':contrast_v,'rgb_contrast':contrast_rgb}})
                                continue
                reason = ('too_small' if area < minimum_area else
                          'oversized_or_possible_joined_parts' if area > 23000 else
                          'rear_clipped' if x+w >= frame.shape[1]-1 else
                          'too_small' if min(w,h) < 16 else
                          'thin_fragment' if max(w,h)/max(1,min(w,h)) > 4.5 else None)
                # A rejected sliver must pass floor and current-boundary
                # checks before its raw connectivity can become an obstacle.
                # Otherwise rails and cable shadows can span the whole tray.
                if reason in ('too_small','thin_fragment') and area >= minimum_area:
                    seed=np.zeros(frame.shape[:2],np.uint8)
                    cv2.drawContours(seed,[contour],-1,1,cv2.FILLED)
                    support_count=int(np.count_nonzero((seed>0)&self.reference_floor))
                    reject_reason=None
                    if support_count < minimum_area:
                        reject_reason='unsupported_reference_floor_seed'
                    elif group.startswith('neutral_'):
                        inner_seed=cv2.erode(seed,np.ones((5,5),np.uint8))>0
                        band_seed=(seed>0)&~inner_seed
                        outer_seed=(cv2.dilate(seed,np.ones((5,5),np.uint8))>0)&(seed==0)&(self.work>0)
                        if band_seed.any() and outer_seed.any():
                            contrast_v=abs(float(np.median(nhsv[:,:,2][band_seed]))-float(np.median(nhsv[:,:,2][outer_seed])))
                            contrast_rgb=float(np.linalg.norm(np.median(normalized[band_seed],axis=0)-np.median(normalized[outer_seed],axis=0)))
                            solidity=area/max(1.,float(cv2.contourArea(cv2.convexHull(contour))))
                            if solidity < .85 and contrast_v < 18 and contrast_rgb < 30:
                                reject_reason='illumination_only_change'
                    if reject_reason:
                        rejected.append({'box_roi':[x,y,x+w,y+h],'group':group,'area':area,
                                         'reason':reject_reason,'reference_floor_support':support_count})
                        continue
                    seed_ids=np.unique(raw_labels[seed>0]);seed_ids=seed_ids[seed_ids!=0]
                    full=np.isin(raw_labels,seed_ids) if len(seed_ids) else np.zeros(raw_labels.shape,bool)
                    if full.any():
                        py,px=np.where(full)
                        if px.min()<x-3 or py.min()<y-3 or px.max()+1>x+w+3 or py.max()+1>y+h+3:
                            blocking.append({'box_roi':[int(px.min()),int(py.min()),int(px.max())+1,int(py.max())+1],
                                             'group':group,'area':area,'reason':'reference_mask_truncates_part',
                                             'partial_box_roi':[x,y,x+w,y+h]})
                if reason:
                    if area > 50:
                        rejected_region={'box_roi': [x, y, x+w, y+h], 'group': group, 'area': area, 'reason': reason}
                        rejected.append(rejected_region)
                        if reason=='too_small' and area>=250 and not group.startswith('neutral_'):
                            rejected_region['_contour']=contour
                        if reason in ('oversized_or_possible_joined_parts','rear_clipped'):
                            blocking.append(dict(rejected_region))
                    continue
                component = np.zeros(frame.shape[:2], np.uint8)
                cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)
                pixels = component > 0
                floor_support = int(np.count_nonzero(pixels & self.reference_floor))
                # Filling the tray contour also fills holes containing fixed
                # rails. A seed must actually replace enough reference floor;
                # only its subsequent connected completion may cross dark gaps.
                if floor_support < minimum_area:
                    rejected.append({'box_roi':[x,y,x+w,y+h],'group':group,'area':area,
                                     'reason':'unsupported_reference_floor_seed',
                                     'reference_floor_support':floor_support})
                    continue
                if group=='dark_part' and np.count_nonzero(pixels & dark_core) < 1200:
                    rejected.append({'box_roi':[x,y,x+w,y+h],'group':group,'area':area,
                                     'reason':'insufficient_very_dark_floor_seed'})
                    continue
                neutral_contrast = None
                current_edge = None
                if group.startswith('neutral_'):
                    if area / max(1,w*h) < .40:
                        rejected.append({'box_roi':[x,y,x+w,y+h],'group':group,'area':area,
                                         'reason':'neutral_shadow_or_fragment'})
                        continue
                    inner = cv2.erode(component,np.ones((5,5),np.uint8)) > 0
                    outer = ((cv2.dilate(component,np.ones((5,5),np.uint8)) > 0) & ~pixels & (self.work > 0))
                    band = pixels & ~inner
                    if band.any() and outer.any():
                        current_v = abs(float(np.median(nhsv[:,:,2][band]))-float(np.median(nhsv[:,:,2][outer])))
                        current_rgb = float(np.linalg.norm(np.median(normalized[band],axis=0)
                                                           -np.median(normalized[outer],axis=0)))
                    else:
                        current_v,current_rgb = 0.,0.
                    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
                    solidity = area/max(1.,hull_area)
                    current_edge={'value_contrast':current_v,'rgb_contrast':current_rgb,
                                  'solidity':solidity}
                    # A vanished highlight can leave a strong difference-image
                    # edge on a currently flat floor. The reverse highlight is
                    # a weak, irregular neutral region; sharp compact
                    # white parts and strong long black/white parts stay valid.
                    flat_current = current_rgb < 4 and current_v < 3
                    weak_highlight = (solidity < .85
                                      and current_rgb < 30 and current_v < 18)
                    if flat_current or weak_highlight:
                        rejected.append({'box_roi':[x,y,x+w,y+h],'group':group,'area':area,
                                         'reason':'illumination_only_change',
                                         'current_boundary':current_edge})
                        continue
                    neutral_contrast = abs(float(np.median(delta_v[inner]))-float(np.median(delta_v[outer]))) if inner.any() and outer.any() else 0.
                    if float(np.median(distance[pixels])) <= .022 and neutral_contrast < 18:
                        region={'box_roi':[x,y,x+w,y+h],'group':group,'area':area,
                                'reason':'neutral_change_shadow_ambiguous','edge_contrast':neutral_contrast}
                        rejected.append(region);blocking.append(dict(region))
                        continue
                merged_mask[pixels] = 255
                bbox = [max(0, x-2), max(0, y-2), min(frame.shape[1], x+w+2), min(frame.shape[0], y+h+2)]
                core_bbox = list(bbox)
                if group == 'cyan_gray':
                    # The gray foam's dark side face changes hue toward green.
                    # Attach only connected, desaturated dark pixels alongside
                    # its existing vertical span; this does not join the bright
                    # saturated green piece stacked below it.
                    side = ((distance > .021) & (delta_v < -45) & (nhsv[:, :, 1] >= 24)
                            & (nhsv[:, :, 1] < 140) & (self.work > 0))
                    # Anchor pale boundary pixels to the green/cyan side
                    # itself. Otherwise a thin neutral shadow can connect the
                    # rear face to a distant object across the floor.
                    colored_side=side & (nhsv[:,:,0]>=60)
                    near_side=cv2.dilate(colored_side.astype(np.uint8),np.ones((15,15),np.uint8))>0
                    side=colored_side | (side & (nhsv[:,:,1]<45) & near_side)
                    angle = abs(float(cv2.minAreaRect(contour)[2])) % 90
                    tilt = min(angle,90-angle)
                    if tilt > 5:
                        side = cv2.morphologyEx(side.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8)) > 0
                    nearby = np.zeros(side.shape,bool)
                    nearby[y:y+h,max(0,x-30):min(frame.shape[1],x+w+80)] = True
                    side = side & nearby
                    # Attach only actual connected dark regions touching the
                    # core within at most two pixels, never a bbox-contained
                    # independent object across a clear gap.
                    _,side_labels,_,_ = cv2.connectedComponentsWithStats(side.astype(np.uint8),connectivity=8)
                    near_core = cv2.dilate(component,np.ones((5,5),np.uint8)) > 0
                    touched = np.unique(side_labels[near_core & side])
                    touched = touched[touched != 0]
                    side = np.isin(side_labels,touched) if len(touched) else np.zeros(side.shape,bool)
                    if tilt > 5:
                        joined = (pixels | side).astype(np.uint8)
                        _,labels,stats,_ = cv2.connectedComponentsWithStats(joined,connectivity=8)
                        counts = np.bincount(labels[pixels].ravel());counts[0] = 0
                        jx,jy,jw,jh,_ = stats[int(np.argmax(counts))]
                        bbox[0] = max(0,min(bbox[0],int(jx)-2))
                        bbox[2] = min(frame.shape[1],max(bbox[2],int(jx+jw)+2))
                    column_support = side[y:y+h].mean(0)
                    scans = () if tilt>5 else ((x+w-2,min(frame.shape[1],x+w+80),1),
                                               (x+1,max(-1,x-30),-1))
                    for start, stop, step in scans:
                        found, gaps, last = False, 0, start
                        for column in range(start,stop,step):
                            if column_support[column] >= .30:
                                found,gaps,last = True,0,column
                            elif found:
                                gaps += 1
                                if gaps >= 2:
                                    break
                            elif abs(column-start)>5:
                                break
                        if found and step>0:
                            bbox[2]=min(frame.shape[1],max(bbox[2],last+3))
                        elif found:
                            bbox[0]=max(0,min(bbox[0],last-2))
                    component[side] = 255
                # Follow only actual same-color connectivity from this seed.
                # The convex envelope fills lighting notches but cannot seed
                # rail/bay objects. Never clip a raw component and authorize
                # the remaining partial center or rear.
                ids = np.unique(raw_labels[pixels]);ids = ids[ids != 0]
                observed = np.isin(raw_labels,ids) if len(ids) else np.zeros(raw_labels.shape,bool)
                if observed.any():
                    uy,ux = np.where(observed)
                    observed_box = [int(ux.min()),int(uy.min()),int(ux.max())+1,int(uy.max())+1]
                    completed_box=[max(0,min(bbox[0],observed_box[0]-2)),
                                   max(0,min(bbox[1],observed_box[1]-2)),
                                   min(frame.shape[1],max(bbox[2],observed_box[2]+2)),
                                   min(frame.shape[0],max(bbox[3],observed_box[3]+2))]
                    completion_reason = ('reference_mask_truncates_part' if np.any(observed & (self.completion_work==0)) else
                                         'oversized_or_possible_joined_parts' if np.count_nonzero(observed)>23000 else
                                         'occluded_or_split_part' if len(ids)!=1 else None)
                    if completion_reason:
                        blocking.append({'box_roi':completed_box,
                                         'group':group,'area':area,'reason':completion_reason,
                                         'partial_box_roi':bbox,'observed_foreground_box_roi':observed_box})
                        continue
                    label_id=int(ids[0])
                    if label_id in raw_owners:
                        previous=raw_owners[label_id]
                        if previous is not None:
                            detections=[item for item in detections if item is not previous]
                            blocking.append({'box_roi':completed_box,'group':group,'area':area,
                                             'reason':'occluded_or_split_part','observed_foreground_box_roi':observed_box})
                            raw_owners[label_id]=None
                        continue
                    component[observed]=255
                    merged_mask[observed]=255
                    bbox=completed_box
                candidate={'box_roi': bbox, 'group': group, 'area': area,
                                   '_component_mask':component,
                                   '_dark_support_mask':observed if group=='dark_part' else None,
                                   '_observed_support_mask':observed,
                                   'core_box_roi':core_bbox,
                                   'median_chroma_difference': float(np.median(distance[pixels])),
                                   'median_value_difference': float(np.median(delta_v[pixels])),
                                   'median_saturation': float(np.median(nhsv[:, :, 1][pixels])),
                                   'neutral_edge_contrast':neutral_contrast,
                                   'current_neutral_boundary':current_edge,
                                   'reference_floor_support':floor_support,
                                   'acceptance_score': 1.0,
                                   'reason': 'very_dark_connected_body' if group=='dark_part' else
                                             'chroma_change_separated_by_hsv' if not group.startswith('neutral_')
                                             else 'neutral_intensity_change'}
                detections.append(candidate)
                if observed.any():
                    raw_owners[label_id]=candidate
        # Dark ownership uses actual low-value pixels, never overlapping boxes.
        # Remove HSV/neutral fragments belonging predominantly to the same dark
        # body. One fragment crossing multiple owners is unresolved, not a join.
        dark_owners=[d for d in detections if d['group']=='dark_part']
        ownership_merges=[]
        dark_removed=set()
        for candidate in detections:
            if candidate['group']=='dark_part' or id(candidate) in dark_removed:
                continue
            component=candidate['_observed_support_mask']
            size=max(1,int(np.count_nonzero(component)))
            overlaps=[(owner,int(np.count_nonzero(component & owner['_dark_support_mask'])))
                      for owner in dark_owners if id(owner) not in dark_removed]
            touched=[(owner,count) for owner,count in overlaps if count>=100]
            split_overlap = sum(count for owner, count in touched if owner['reason'] == SPLIT_REASON)
            if split_overlap >= .95 * size:
                # The neutral/color pass saw the same pixels as the dark split.
                # Preserve the verified seam when discarding this duplicate.
                dark_removed.add(id(candidate))
                continue
            if touched:
                regions=[candidate]+[owner for owner,_ in touched]
                # A moderately lit stripe can split a black cylinder's deep
                # shadows. Reconnect only within the observed body's hull;
                # a brighter floor gap still separates independent objects.
                union_mask=np.zeros(component.shape,np.uint8)
                for region in regions:
                    union_mask[region['_observed_support_mask']]=1
                hull=np.zeros_like(union_mask)
                cv2.fillConvexPoly(hull,cv2.convexHull(cv2.findNonZero(union_mask)),1)
                lit_body=((hull>0)&(nhsv[:,:,2]<65)&(delta_v < -50)).astype(np.uint8)
                _,body_labels,body_stats,_=cv2.connectedComponentsWithStats(lit_body,connectivity=8)
                body_ids=[]
                for owner,_ in touched:
                    counts=np.bincount(body_labels[owner['_dark_support_mask']]);counts[0]=0
                    body_ids.append(int(np.argmax(counts)))
                if (not (len(touched) > 1 and split_overlap)
                        and body_ids[0]>0 and len(set(body_ids))==1
                        and body_stats[body_ids[0],cv2.CC_STAT_AREA]/max(1,np.count_nonzero(hull))>=.85):
                    body=body_labels==body_ids[0]
                    by,bx=np.where(body)
                    owner=touched[0][0]
                    owner['box_roi']=[max(0,int(bx.min())-2),max(0,int(by.min())-2),
                                      min(frame.shape[1],int(bx.max())+3),min(frame.shape[0],int(by.max())+3)]
                    owner['_component_mask'][body]=255
                    owner['_dark_support_mask']=body
                    owner['_observed_support_mask']=body
                    dark_removed.update(id(item) for item in regions if item is not owner)
                    ownership_merges.append({'from_group':candidate['group'],'into_group':'dark_part',
                                             'reason':'connected_lit_dark_body'})
                    continue
            if len(touched)>1:
                boxes=[item['box_roi'] for item in regions]
                blocking.append({'box_roi':[min(b[0] for b in boxes),min(b[1] for b in boxes),
                                             max(b[2] for b in boxes),max(b[3] for b in boxes)],
                                 'group':'dark_part','reason':'occluded_or_split_part'})
                dark_removed.update(id(item) for item in regions)
            elif touched and touched[0][1]/size>=.5:
                owner,count=touched[0]
                dark_removed.add(id(candidate))
                ownership_merges.append({'from_group':candidate['group'],'into_group':'dark_part',
                                         'overlap_pixels':count,'fragment_pixels':size,
                                         'reason':'same_actual_dark_body'})
        detections=[item for item in detections if id(item) not in dark_removed]
        # Small, vertically aligned color slivers can be the dark side face of
        # one cube. Equal-size adjacent color regions remain separate parts.
        merges = []
        for small in sorted(list(detections), key=lambda d: d['area']):
            if not any(small is item for item in detections):
                continue
            sx1, sy1, sx2, sy2 = small['box_roi']
            for large in sorted(detections, key=lambda d: -d['area']):
                if large is small or small['area'] > large['area']*.5:
                    continue
                dark_side = (large['group']=='cyan_gray' and small['group']=='green'
                             and small['median_saturation'] < 120
                             and small['median_value_difference'] < large['median_value_difference']-10)
                actual_contact = np.any((cv2.dilate(large['_component_mask'],np.ones((5,5),np.uint8))>0)
                                        & (small['_component_mask']>0))
                if not dark_side or not actual_contact:
                    continue
                lx1, ly1, lx2, ly2 = large['box_roi']
                if sy1 < large['core_box_roi'][1]-6 or sy2 > large['core_box_roi'][3]+6:
                    continue
                ix = max(0,min(sx2,lx2)-max(sx1,lx1))
                iy = max(0,min(sy2,ly2)-max(sy1,ly1))
                contained = ix*iy / max(1,(sx2-sx1)*(sy2-sy1)) > .6
                side_face = (ix / max(1,sx2-sx1) > .2 and iy / max(1,sy2-sy1) > .65
                             and lx1-15 <= (sx1+sx2)/2 <= lx2+15
                             and ly1-15 <= (sy1+sy2)/2 <= ly2+15)
                if not (contained or side_face):
                    continue
                large['box_roi'] = [min(sx1,lx1),min(sy1,ly1),max(sx2,lx2),max(sy2,ly2)]
                large['_component_mask'] |= small['_component_mask']
                large.setdefault('merged_side_regions',[]).append({'group':small['group'],'box_roi':small['box_roi']})
                merges.append({'into_group':large['group'],'from_group':small['group'],'reason':'small_contained_or_aligned_side_face'})
                detections = [item for item in detections if item is not small]
                break
        # A nearby same-color piece filtered by area may be an occluded portion
        # of the main part. Record the whole uncertainty as blocking; unlike
        # dark-face merging, this never authorizes a combined action box.
        blocked_ids = set()
        fragments = [r for r in rejected if r['reason']=='too_small' and r['area']>=250
                     and not r['group'].startswith('neutral_')]
        for fragment in fragments:
            fx1,fy1,fx2,fy2 = fragment['box_roi']
            for owner in list(blocking)+list(detections):
                if owner.get('group') != fragment['group'] or fragment['area']>owner.get('area',0)*.4:
                    continue
                ox1,oy1,ox2,oy2 = owner['box_roi']
                if ('_component_mask' in owner and '_contour' in fragment
                        and ox1<=fx1 and oy1<=fy1 and ox2>=fx2 and oy2>=fy2):
                    hull=np.zeros(frame.shape[:2],np.uint8)
                    cv2.fillConvexPoly(hull,cv2.convexHull(cv2.findNonZero(owner['_component_mask'])),1)
                    fragment_mask=np.zeros_like(hull)
                    cv2.drawContours(fragment_mask,[fragment['_contour']],-1,1,cv2.FILLED)
                    if np.count_nonzero(fragment_mask & hull)/max(1,np.count_nonzero(fragment_mask))>=.98:
                        continue
                ix=max(0,min(fx2,ox2)-max(fx1,ox1));iy=max(0,min(fy2,oy2)-max(fy1,oy1))
                gap_x=max(0,fx1-ox2,ox1-fx2);gap_y=max(0,fy1-oy2,oy1-fy2)
                if not ((ix/max(1,fx2-fx1)>=.25 and gap_y<=8)
                        or (iy/max(1,fy2-fy1)>=.5 and gap_x<=4)):
                    continue
                union=[min(fx1,ox1),min(fy1,oy1),max(fx2,ox2),max(fy2,oy2)]
                if any(owner is b for b in blocking):
                    owner['box_roi']=union
                    owner.setdefault('split_fragment_boxes',[]).append(fragment['box_roi'])
                else:
                    blocking.append({'box_roi':union,'group':owner['group'],'area':owner['area'],
                                     'reason':'occluded_or_split_part','partial_box_roi':owner['box_roi'],
                                     'split_fragment_boxes':[fragment['box_roi']]})
                    blocked_ids.add(id(owner))
                break
        detections = [item for item in detections if id(item) not in blocked_ids]
        for region in rejected:
            region.pop('_contour',None)
        accepted = []
        for candidate in detections:
            # Pass observed geometry to the contact planner; a bounding box
            # alone cannot prove that a 25 mm plate has room to descend.
            mask=candidate['_component_mask'].copy()
            bx1,by1,bx2,by2=candidate['box_roi']
            mask[:by1]=0;mask[by2:]=0;mask[:,:bx1]=0;mask[:,bx2:]=0
            points=cv2.findNonZero(mask)
            if points is not None:
                candidate['footprint_roi']=cv2.convexHull(points).reshape(-1,2).tolist()
            candidate.pop('_component_mask',None)
            candidate.pop('_dark_support_mask',None)
            candidate.pop('_observed_support_mask',None)
            if candidate['box_roi'][2] >= frame.shape[1]:
                blocking.append({'box_roi':candidate['box_roi'],'reason':'rear_clipped','group':candidate['group']})
            else:
                accepted.append(candidate)
        from tray_pale import supplement_pale
        accepted, pale_diagnostics = supplement_pale(normalized,nhsv,distance,self.work,self.completion_work,
                                                     self.pale_hints,accepted,blocking)
        return accepted, {'background_gain_bgr': gain.tolist(), 'rejected': rejected, 'side_merges':merges,
                          'pale_diagnostics':pale_diagnostics,
                          'excluded_servo_cables':cable_regions,
                          'dark_ownership_merges':ownership_merges,
                          'blocking_regions':blocking}, merged_mask

    def detect(self, frame):
        _validate_image(frame, 'frame')
        if frame.shape != self.ref.shape:
            raise ValueError('foreground frame must match the saved reference shape')
        detections, diagnostics, mask = self._detect_diagnostics(frame)
        self.last_diagnostics = dict(diagnostics, foreground_mask=mask, candidates=detections,
                                     reference_sha256=self._reference_sha256,
                                     status=diagnostics.get('reason', 'ok'),
                                     score_semantics='rule_acceptance_not_model_probability')
        return [tuple(d['box_roi'])+(d['acceptance_score'],) for d in detections]
