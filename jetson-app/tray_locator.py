import math

import numpy as np

from motion_limits import X_MIN_MM, Y_MIN_MM, X_MAX_MM, Y_MAX_MM, position_in_machine


def _confidence_threshold(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value)) or not 0 <= value <= 1):
        raise ValueError('confidence threshold must be a finite number between zero and one')
    return float(value)


def _compensation_axes(settings):
    try:
        center_x, center_y = map(float, settings['center'])
        span_x, span_y = map(float, settings['span'])
        left, right, top, bottom = (float(settings[key]) for key in
                                    ('x_left', 'x_right', 'y_top', 'y_bottom'))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid tray position compensation') from error
    if (not all(math.isfinite(value) for value in (center_x, center_y, span_x, span_y,
                                                  left, right, top, bottom))
            or span_x <= 0 or span_y <= 0):
        raise ValueError('tray position compensation requires finite values and positive spans')
    return (center_x, span_x, left, right), (center_y, span_y, top, bottom)


def _axis_correction(position, center, span, lower_side, upper_side):
    distance = position - center
    factor = min(1.0, abs(distance) / span)
    return factor * (lower_side if distance < 0 else upper_side)


class TrayLocator:
    """Select the first part along negative X and start behind its rear edge.

    Optional compensation uses millimetres throughout. The Windows settings
    expose the center, span and each side's endpoint adjustment. Since the
    compiled calculation is unavailable, the Linux profile uses a linear ramp
    from zero at center to the configured adjustment at one span and beyond.
    The ramp uses the calibrated rear X and center Y before the tool offsets.
    Projecting all four image-box corners supports reversed/rotated camera
    mappings. Target-only motion caps unreachable starts at machine limits.
    If no contact lane is available, the recorded boundary fallback keeps
    travel limits but does not guarantee object clearance.
    With skip_unsafe, only candidates with safe centers and corrected targets
    enter left-first selection. Optional selection_region bounds candidate
    centers in full-frame pixels independently of the detector's crop.
    Optional mechanical-X column boundaries select earlier columns first,
    then increasing Y within each column; omitted boundaries keep strict X.
    """

    def __init__(self, calibration, detector, roi, offsets=(4,0), confidence=0.35, *,
                 compensation=None, skip_unsafe=False, selection_region=None,
                 column_boundaries=None, motion_target_only=False, pusher_width_mm=None,
                 contact_ignore_unresolved=False, contact_side_clearance_mm=0.5,
                 contact_allow_sweep_mm2=0.25):
        if type(motion_target_only) is not bool:
            raise ValueError('motion_target_only must be a boolean')
        if type(contact_ignore_unresolved) is not bool:
            raise ValueError('contact_ignore_unresolved must be a boolean')
        self.calibration, self.detector = calibration, detector
        self.roi, self.offsets, self.confidence = roi, offsets, _confidence_threshold(confidence)
        self.motion_target_only = motion_target_only
        self.contact_ignore_unresolved = contact_ignore_unresolved
        if contact_ignore_unresolved and not motion_target_only:
            raise ValueError('contact relaxation requires target-only motion validation')
        for name, value, upper in (('contact_side_clearance_mm', contact_side_clearance_mm, 5),
                                   ('contact_allow_sweep_mm2', contact_allow_sweep_mm2, 500)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= upper):
                raise ValueError('%s must be a finite number between zero and %g' % (name, upper))
        self.contact_side_clearance_mm = float(contact_side_clearance_mm)
        self.contact_allow_sweep_mm2 = float(contact_allow_sweep_mm2)
        if pusher_width_mm is not None:
            if (isinstance(pusher_width_mm,bool) or not isinstance(pusher_width_mm,(int,float))
                    or not math.isfinite(pusher_width_mm) or pusher_width_mm<=0 or not motion_target_only):
                raise ValueError('pusher width requires positive millimetres and target-only motion validation')
        self.pusher_width_mm=pusher_width_mm
        self.last_diagnostics = {}
        self._compensation = _compensation_axes(compensation) if compensation is not None else None
        self.skip_unsafe = skip_unsafe
        self.selection_region = None
        if selection_region is not None:
            try:
                region = np.asarray(selection_region, dtype=float)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError('invalid tray selection region') from error
            if (region.shape != (4,) or not np.isfinite(region).all()
                    or region[0] >= region[2] or region[1] >= region[3]):
                raise ValueError('invalid tray selection region')
            self.selection_region = tuple(float(value) for value in region)
        self.column_boundaries = None
        if column_boundaries is not None:
            try:
                boundaries = np.asarray(column_boundaries, dtype=float)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError('invalid tray column boundaries') from error
            if (boundaries.shape != (2,) or not np.isfinite(boundaries).all()
                    or boundaries[0] >= boundaries[1]):
                raise ValueError('invalid tray column boundaries')
            self.column_boundaries = tuple(float(value) for value in boundaries)

    def _selection_key(self, center):
        if self.column_boundaries is None:
            return tuple(center)
        return (sum(center[0] >= boundary for boundary in self.column_boundaries),
                center[1], center[0])

    @staticmethod
    def _clamp_motion_target(target):
        if len(target) != 2 or not all(math.isfinite(value) for value in target):
            raise ValueError('motion target must contain two finite coordinates')
        return (min(max(float(target[0]), X_MIN_MM), X_MAX_MM),
                min(max(float(target[1]), Y_MIN_MM), Y_MAX_MM))

    def _boundary_push_target(self, trace):
        records = [record for record in trace['candidates']
                   if 'selection_key' in record and record.get('proposed_target_mm') is not None
                   and record.get('rejection_stage') in (None, 'target_range')]
        if not records:
            records = [record for record in trace['candidates']
                       if record.get('rejection_stage') == 'foreground_ambiguity'
                       and 'selection_key' in record]
        for record in sorted(records, key=lambda item: tuple(item['selection_key'])):
            try:
                proposed = record.get('proposed_target_mm')
                if proposed is None:
                    left, top, right, bottom = record['box_full']
                    corners = [self.calibration.project_pixel(px, py)
                               for px in (left, right) for py in (top, bottom)]
                    rear = (max(point[0] for point in corners), record['center_mm'][1])
                    proposed = self._propose_push_target(rear)
                if len(proposed) != 2 or any(value is None for value in proposed):
                    continue
                if not all(math.isfinite(value) for value in proposed) or proposed[0] <= 10:
                    continue
                target = self._clamp_motion_target(proposed)
                if target == tuple(proposed):
                    target = (X_MAX_MM, target[1])
                self.calibration.validate_physical(*target)
                if not position_in_machine(*target) or float('%.3f' % target[0]) <= 10:
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            record.update(selected=True, target_mm=list(target), contact_target_mm=list(target),
                          forced_boundary_push=True, contact_reason='forced_boundary_target',
                          nominal_rejection_stage=record.get('rejection_stage'),
                          nominal_reason=record.get('reason'), rejection_stage=None, reason=None)
            trace.setdefault('contact_planning', {}).update(
                policy='forced_boundary_after_contact_failure', bypassed_clearance_check=True)
            trace.update(selection_status='selected_forced_boundary', selected_index=record['index'],
                         target_mm=list(target), hold_reason=None, blocked_index=None)
            return target
        return None

    def locate(self, frame):
        x,y,w,h = self.roi
        trace = self.last_diagnostics = {
            'roi_xywh': list(self.roi), 'confidence_threshold': self.confidence,
            'selection_region': self.selection_region,
            'column_boundaries': self.column_boundaries,
            'motion_target_only': self.motion_target_only,
            'candidates': [], 'selected_index': None, 'target_mm': None,
            'selection_status': 'no_safe_candidate', 'hold_reason': None,
            'blocked_index': None,
        }
        if any(int(v) != v for v in self.roi) or x < 0 or y < 0 or w <= 0 or h <= 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
            raise ValueError('invalid ROI')
        candidates = []
        blocked = []
        for index, detection in enumerate(self.detector.detect(frame[y:y+h,x:x+w])):
            x1,y1,x2,y2,score = map(float,detection)
            record = {'index': index,
                      'box_roi': [v if math.isfinite(v) else None for v in (x1,y1,x2,y2)],
                      'confidence': score if math.isfinite(score) else None,
                      'rejection_stage': None, 'reason': None, 'selected': False}
            trace['candidates'].append(record)
            if not np.isfinite([x1,y1,x2,y2,score]).all():
                record.update(rejection_stage='decoded_values', reason='nonfinite box or score')
                continue
            if score < self.confidence:
                record.update(rejection_stage='confidence', reason='below candidate threshold')
                continue
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                record.update(rejection_stage='bbox', reason='invalid box or outside detector ROI')
                continue
            pixel_x, pixel_y = x + (x1 + x2) / 2, y + (y1 + y2) / 2
            record['box_full'] = [x+x1,y+y1,x+x2,y+y2]
            record['center_pixel'] = [pixel_x,pixel_y]
            if self.selection_region is not None:
                left, top, right, bottom = self.selection_region
                if not (left <= pixel_x <= right and top <= pixel_y <= bottom):
                    record.update(rejection_stage='selection_region', reason='center outside selection region')
                    continue
            stage = 'center_projection' if self.motion_target_only else 'center_range'
            try:
                center = (self.calibration.project_pixel(pixel_x, pixel_y) if self.motion_target_only
                          else self.calibration.map_pixel(pixel_x, pixel_y))
                record['center_mm'] = list(center)
                record['selection_key'] = list(self._selection_key(center))
                # Box corners describe object geometry, not commanded positions.
                # A corner can be outside the usable Y strip while its center
                # and the final push target remain valid.
                stage = 'box_projection'
                corners = [self.calibration.project_pixel(x + px, y + py)
                           for px in (x1, x2) for py in (y1, y2)]
                rear = (max(point[0] for point in corners), center[1])
                record['front_x_mm'] = min(point[0] for point in corners)
                record['rear_mm'] = list(rear)
                stage = 'target_range'
                if self.motion_target_only:
                    proposed = self._propose_push_target(rear)
                    record['proposed_target_mm'] = [v if math.isfinite(v) else None for v in proposed]
                target = self._push_target(rear) if self.skip_unsafe or self.motion_target_only else rear
                if self.skip_unsafe or self.motion_target_only:
                    record['target_mm'] = list(target)
            except ValueError as error:
                record.update(rejection_stage=stage, reason=str(error))
                if self.motion_target_only and 'selection_key' in record:
                    blocked.append(record)
                if self.skip_unsafe or self.motion_target_only:
                    continue
                raise
            candidates.append((center, target, record))
        if self.motion_target_only:
            detector_trace = getattr(self.detector, 'last_diagnostics', None) or {}
            if not isinstance(detector_trace, dict):
                raise ValueError('invalid foreground diagnostics')
            if detector_trace.get('hold_reason'):
                trace.update(selection_status='blocked_foreground_reference',
                             hold_reason=str(detector_trace['hold_reason']))
                return None
            for region in detector_trace.get('blocking_regions', ()):
                try:
                    values = np.asarray(region['box_roi'], dtype=float)
                    if (values.shape != (4,) or not np.isfinite(values).all()
                            or not (0 <= values[0] < values[2] <= w
                                    and 0 <= values[1] < values[3] <= h)):
                        raise ValueError('invalid foreground blocking box')
                    bx1, by1, bx2, by2 = (float(value) for value in values)
                    center = self.calibration.project_pixel(x+(bx1+bx2)/2, y+(by1+by2)/2)
                    corners = [self.calibration.project_pixel(x+px, y+py)
                               for px in (bx1,bx2) for py in (by1,by2)]
                except (KeyError, TypeError, ValueError, OverflowError) as error:
                    raise ValueError('invalid foreground blocking region') from error
                # An unresolved region may contain several parts. Its earliest
                # corner takes priority rather than silently hiding a left part
                # behind the combined region's center in a later column.
                priority = (min(point[0] for point in corners), min(point[1] for point in corners))
                record = {'index': len(trace['candidates']),
                          'box_roi': [bx1,by1,bx2,by2],
                          'box_full': [x+bx1,y+by1,x+bx2,y+by2],
                          'center_mm': list(center), 'selection_key': list(self._selection_key(priority)),
                          'priority_basis': 'earliest_corner_of_unresolved_region',
                          'confidence': None, 'selected': False,
                          'rejection_stage': 'foreground_ambiguity',
                          'reason': str(region.get('reason', 'unresolved_foreground'))}
                trace['candidates'].append(record)
                blocked.append(record)
        earliest_blocked = min(blocked, key=lambda item: item['selection_key']) if blocked else None
        if self.pusher_width_mm is not None:
            ambiguous=[r for r in blocked if r['rejection_stage']=='foreground_ambiguity']
            first_ambiguous=min(ambiguous,key=lambda r:r['selection_key']) if ambiguous else None
            known=[r for r in trace['candidates'] if r.get('proposed_target_mm') is not None]
            if first_ambiguous is not None and (not known or
                    tuple(first_ambiguous['selection_key'])<=min(tuple(r['selection_key']) for r in known)):
                if not self.contact_ignore_unresolved:
                    trace.update(selection_status='blocked_earlier_candidate',blocked_index=first_ambiguous['index'],
                                 hold_reason=self._blocked_reason(first_ambiguous))
                    return self._boundary_push_target(trace)
                # Operator-selected field option: an unresolved region keeps its
                # place in the ordering but stops vetoing every located part
                # behind it. The region is still reported in the trace and in
                # _locate_with_contacts, where it can also still block a lane.
                trace['contact_planning']={'width_mm':self.pusher_width_mm,
                                           'policy':'clear_plate_and_object_sweep'}
                trace['contact_planning']['earlier_unresolved_index']=first_ambiguous['index']
                trace['contact_planning']['earlier_unresolved_reason']=first_ambiguous.get('reason')
            target = self._locate_with_contacts(trace,detector_trace,
                None if self.contact_ignore_unresolved else first_ambiguous)
            if target is None:
                return self._boundary_push_target(trace)
            return target
        if not candidates:
            if earliest_blocked is not None:
                trace.update(selection_status='blocked_earlier_candidate', blocked_index=earliest_blocked['index'],
                             hold_reason=self._blocked_reason(earliest_blocked))
            return None
        center, target, selected = min(candidates, key=lambda item: self._selection_key(item[0]))
        if (earliest_blocked is not None
                and tuple(earliest_blocked['selection_key']) <= self._selection_key(center)):
            trace.update(selection_status='blocked_earlier_candidate', blocked_index=earliest_blocked['index'],
                         hold_reason=self._blocked_reason(earliest_blocked))
            return None
        result = target if self.skip_unsafe or self.motion_target_only else self._push_target(target)
        selected.update(selected=True, target_mm=list(result))
        trace.update(selection_status='selected', selected_index=selected['index'], target_mm=list(result))
        return result

    def _locate_with_contacts(self,trace,detector_trace,first_ambiguous):
        from tray_contact import plan_push_contact
        trace.setdefault('contact_planning',{}).update(width_mm=self.pusher_width_mm,
                                                       policy='clear_plate_and_object_sweep')
        observations=detector_trace.get('candidates',[])
        if not trace['candidates']:
            return None
        polygons={}
        x,y,w,h=self.roi
        try:
            for record in trace['candidates']:
                index=record['index']
                if record.get('rejection_stage')=='foreground_ambiguity':
                    bx1,by1,bx2,by2=record['box_roi']
                    points=((bx1,by1),(bx2,by1),(bx2,by2),(bx1,by2))
                else:
                    points=observations[index]['footprint_roi']
                points=np.asarray(points,dtype=float)
                if (points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all()
                        or np.any(points[:,0]<0) or np.any(points[:,0]>w)
                        or np.any(points[:,1]<0) or np.any(points[:,1]>h)):
                    raise ValueError('invalid foreground footprint')
                polygons[index]=[self.calibration.project_pixel(x+px,y+py) for px,py in points]
        except (KeyError,IndexError,TypeError,ValueError) as error:
            trace.update(selection_status='blocked_contact_geometry',
                         hold_reason='缺少完整物块轮廓，暂不生成贴边推送坐标',contact_error=str(error))
            return None
        unresolved=(set() if not self.contact_ignore_unresolved else
                    {record['index'] for record in trace['candidates']
                     if record.get('rejection_stage')=='foreground_ambiguity'} & set(polygons))
        if unresolved:
            # Operator-selected field option: an unresolved region keeps its
            # place in the earlier-candidate ordering above, but it no longer
            # vetoes a lane that is proven clear against every part the
            # detector actually located. The trace records that reliance so a
            # push only the relaxed rule allows stays visible afterwards.
            trace['contact_planning']['ignored_unresolved_regions']=sorted(unresolved)

        def lane_obstacles(record_index):
            return [polygon for index,polygon in polygons.items()
                    if index!=record_index and index not in unresolved]

        choices=[]
        for record in trace['candidates']:
            preferred=record.get('proposed_target_mm')
            if preferred is None or any(v is None for v in preferred):
                continue
            if first_ambiguous is not None and tuple(record['selection_key'])>=tuple(first_ambiguous['selection_key']):
                continue
            target=plan_push_contact(polygons[record['index']],
                lane_obstacles(record['index']),preferred,
                self.calibration.physical_region,width_mm=self.pusher_width_mm,
                gap_mm=max(0,self.offsets[0]),
                side_clearance_mm=self.contact_side_clearance_mm,
                sweep_tolerance_mm2=self.contact_allow_sweep_mm2)
            if target is None:
                record['contact_reason']='no_clear_plate_and_object_path'
                continue
            self.calibration.validate_physical(*target)
            if not position_in_machine(*target) or float('%.3f'%target[0])<=10:
                raise ValueError('contact planner produced an unsafe motion target')
            record['contact_target_mm']=list(target)
            choices.append((tuple(record['selection_key']),target,record))
        if not choices:
            trace.update(selection_status='blocked_contact_clearance',
                         hold_reason='当前没有满足 %.0f mm 推板的可达无碰撞通道，等待场景变化'%self.pusher_width_mm)
            return None
        _,target,selected=min(choices,key=lambda item:item[0])
        if unresolved:
            conservative=plan_push_contact(polygons[selected['index']],
                [p for index,p in polygons.items() if index!=selected['index']],
                selected['proposed_target_mm'],self.calibration.physical_region,
                width_mm=self.pusher_width_mm,gap_mm=max(0,self.offsets[0]),
                side_clearance_mm=.5,sweep_tolerance_mm2=.25)
            if conservative is None:
                selected['relied_on_ignored_regions']=sorted(unresolved)
                trace['contact_planning']['selected_relied_on_ignored_regions']=sorted(unresolved)
        selected.update(selected=True,target_mm=list(target),
                        nominal_rejection_stage=selected['rejection_stage'],nominal_reason=selected['reason'],
                        rejection_stage=None,reason=None,
                        contact_adjustment_y_mm=target[1]-selected['proposed_target_mm'][1])
        trace.update(selection_status='selected',selected_index=selected['index'],target_mm=list(target))
        return target

    def _blocked_reason(self, record):
        if record.get('rejection_stage') == 'foreground_ambiguity':
            if record.get('reason') in ('rear_clipped', 'neutral_part_clipped_at_roi_edge'):
                return '前排物块后缘未完整入镜，请向托盘内侧调整该物块'
            if record.get('reason') in ('reference_mask_truncates_part', 'occluded_or_split_part'):
                return '前排物块轮廓不完整，请移开遮挡并向托盘内侧调整'
            return '前排物块可能相连或轮廓不明确，请分开摆放后继续'
        proposed = record.get('proposed_target_mm')
        if proposed is not None and all(value is not None for value in proposed):
            return ('前排物块起推点超出范围（X=%.1f，Y=%.1f mm），请向托盘内侧调整该物块' % tuple(proposed))
        return '前排物块暂无法安全定位，请向托盘内侧调整该物块'

    def _propose_push_target(self, rear):
        edge_x, edge_y = rear
        minimum_x = edge_x + max(self.offsets[0], 0)
        if self._compensation is not None:
            x_axis, y_axis = self._compensation
            edge_x += _axis_correction(edge_x, *x_axis)
            edge_y += _axis_correction(edge_y, *y_axis)
        return (max(minimum_x, edge_x + self.offsets[0]), edge_y + self.offsets[1])

    def _push_target(self, rear):
        result = self._propose_push_target(rear)
        if self.motion_target_only:
            result = self._clamp_motion_target(result)
        self.calibration.validate_physical(*result)
        if self.motion_target_only and (not position_in_machine(*result) or float('%.3f' % result[0]) <= 10):
            raise ValueError('起推点须在 X 0–{:g}、Y 0–{:g} mm 内，且发送坐标 X 必须大于 10 mm'.format(X_MAX_MM,Y_MAX_MM))
        return result


class TrayDetector:
    """Decode one-class tray detections from a YOLO backend on an ROI."""
    def __init__(self, backend, confidence=0.35, iou=0.5, *, feature_input=False):
        self.backend, self.confidence, self.iou = backend, _confidence_threshold(confidence), iou
        self.feature_input = feature_input

    def detect(self, frame):
        from vision import letterbox_rgb, decode_yolo_raw
        if self.feature_input:
            from feature_preprocess import feature_bgr
            frame = feature_bgr(frame)
        tensor, transform = letterbox_rgb(frame, size=640)
        detections = decode_yolo_raw(self.backend.infer(tensor), transform, ['object'],
                                     self.confidence, self.iou, expected_count=1)
        return [tuple(d.box) + (d.confidence,) for d in detections]
