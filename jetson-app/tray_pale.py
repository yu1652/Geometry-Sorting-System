"""Model-guided recovery of pale plastic on the calibrated neutral tray.

Hints never authorize motion alone. A separate, whole current-image component
must agree with one hint. The conservative union rectangle is passed downstream
as occupied space; connected/uncertain components remain obstacles.
"""
import cv2
import numpy as np


def _local_boundary_contrast(image, component, work):
    """Compare nearby pixels across each edge, without cancelling opposite faces."""
    values=[]
    for dx,dy in ((6,0),(-6,0),(0,6),(0,-6)):
        outside=~np.roll(component,(dy,dx),(0,1))
        valid=component & outside & (work>0) & np.roll(work>0,(dy,dx),(0,1))
        # np.roll wraps; border pixels must never supply an opposite-edge pair.
        valid[:6]=False;valid[-6:]=False;valid[:,:6]=False;valid[:,-6:]=False
        other=np.roll(image,(dy,dx),(0,1))
        values.extend(np.linalg.norm(image[valid]-other[valid],axis=1).tolist())
    return len(values)>=100 and np.median(values)>=6 and np.percentile(values,75)>=12


def supplement_pale(normalized, hsv, distance, work, completion_work,
                    hints, candidates, blocking):
    height, width = hsv.shape[:2]
    valid = []
    for hint in hints:
        try:
            values = np.asarray(hint, dtype=float)
            if values.shape != (5,) or not np.isfinite(values).all():
                continue
            x1,y1,x2,y2,score = values.tolist()
            if not .4 <= score <= 1 or not (0<=x1<x2<=width and 0<=y1<y2<=height):
                continue
            if min(x2-x1,y2-y1)<25:
                continue
            valid.append([x1,y1,x2,y2,score])
        except (TypeError,ValueError):
            continue
    if not valid:
        return candidates, []
    # This camera's white plastic has a slight cool cast relative to its
    # saved tray. Require that actual chroma change as well as low saturation;
    # bright neutral LED patches alone cannot seed a white part.
    quantized = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR).astype(np.float32)
    blue,green,red = [quantized[:,:,i] for i in range(3)]
    mask = ((hsv[:,:,1]<80) & (hsv[:,:,2]>90) & (green-red>4)
            & (blue-red>0) & (distance>.012)).astype(np.uint8)
    # Thin floor reflections can bridge into a white face. Remove sub-7px
    # strands before connectivity; accepted bodies must still be at least 25px
    # wide and the full model box plus 3px margin remains occupied geometry.
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(mask,connectivity=8)
    recovered, diagnostics = [], []
    for label in range(1,count):
        x,y,w,h,area = map(int,stats[label])
        if area<1500 or min(w,h)<25:
            continue
        component = labels==label
        matches = []
        for hint in valid:
            x1,y1,x2,y2,_ = hint
            ix1,iy1,ix2,iy2 = int(x1),int(y1),int(np.ceil(x2)),int(np.ceil(y2))
            covered = np.count_nonzero(component[iy1:iy2,ix1:ix2])
            if covered/max(1,(x2-x1)*(y2-y1)) >= .35 and covered/area >= .25:
                matches.append(hint)
        if not matches or np.count_nonzero(component & (work>0)) < max(1200,.2*area):
            continue
        contour=max(cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2],key=cv2.contourArea)
        solidity=cv2.contourArea(contour)/max(1,cv2.contourArea(cv2.convexHull(contour)))
        box=[x,y,x+w,y+h]
        reason=None
        if (x<=0 or y<=0 or x+w>=width or y+h>=height or
                any(a<=0 or b<=0 or c>=width or d>=height for a,b,c,d,s in matches)):
            reason='occluded_or_split_part'
        elif len(matches)!=1 or area>18000 or solidity<.8 or max(w,h)/min(w,h)>3:
            reason='pale_model_component_ambiguous'
        elif np.any(component & (completion_work==0)):
            reason='reference_mask_truncates_part'
        else:
            hx1,hy1,hx2,hy2,_=matches[0]
            if (max(hx1-x,hy1-y,x+w-hx2,y+h-hy2)>18 or
                    max(x-hx1,y-hy1,hx2-x-w,hy2-y-h)>30):
                reason='pale_model_component_ambiguous'
        if reason:
            for hint in matches:
                box=[min(box[0],int(hint[0])),min(box[1],int(hint[1])),
                     max(box[2],int(np.ceil(hint[2]))),max(box[3],int(np.ceil(hint[3])))]
            blocking.append({'box_roi':box,'group':'model_pale','reason':reason})
            diagnostics.append({'box_roi':box,'reason':reason})
            continue
        body=component.astype(np.uint8)
        inner=component & (cv2.erode(body,np.ones((5,5),np.uint8))==0)
        outer=(cv2.dilate(body,np.ones((9,9),np.uint8))>0) & ~component & (work>0)
        if np.count_nonzero(outer)<100 or (np.linalg.norm(
                np.median(normalized[inner],axis=0)-np.median(normalized[outer],axis=0))<6
                and not _local_boundary_contrast(normalized,component,work)):
            diagnostics.append({'box_roi':box,'reason':'pale_weak_current_boundary'})
            continue
        hint=matches[0]
        # Cover both the visible body and the model's full rear face. A pale
        # interior patch must not become a shortened push-start coordinate.
        box=[max(0,min(x,int(hint[0]))-3),max(0,min(y,int(hint[1]))-3),
             min(width,max(x+w,int(np.ceil(hint[2])))+3),
             min(height,max(y+h,int(np.ceil(hint[3])))+3)]
        if box[2]>=width:
            blocking.append({'box_roi':box,'group':'model_pale','reason':'rear_clipped'})
            continue
        bx1,by1,bx2,by2=box
        # Only replace a fully contained fragment supported by the
        # same observed component. Existing ambiguity obstacles are preserved.
        kept=[]
        for candidate in candidates:
            cx1,cy1,cx2,cy2=candidate['box_roi']
            same=(bx1<=cx1 and by1<=cy1
                  and bx2>=cx2 and by2>=cy2 and
                  np.count_nonzero(component[cy1:cy2,cx1:cx2])/max(1,(cx2-cx1)*(cy2-cy1))>.5)
            if not same:
                kept.append(candidate)
        candidates=kept
        recovered.append({'box_roi':box,'group':'model_pale','area':area,
                          'acceptance_score':1.,'reason':'model_and_whole_pale_component',
                          'model_score':hint[4],'model_box_roi':hint[:4],
                          'footprint_roi':[[bx1,by1],[bx2,by1],[bx2,by2],[bx1,by2]]})
    return candidates+recovered, diagnostics
