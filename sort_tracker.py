import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def _iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    return wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh + 1e-9
    )

class _KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.x[:4] = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2,
                                  (bbox[2]-bbox[0])*(bbox[3]-bbox[1]), 1.0]).reshape((4,1))
        self.time_since_update = 0
        _KalmanBoxTracker.count += 1
        self.id = _KalmanBoxTracker.count
        self.hit_streak = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return np.array([self.kf.x[0,0]-self.kf.x[2,0]/2, self.kf.x[1,0]-self.kf.x[2,0]/2,
                         self.kf.x[0,0]+self.kf.x[2,0]/2, self.kf.x[1,0]+self.kf.x[2,0]/2]).reshape((1,4))

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2,
                                 (bbox[2]-bbox[0])*(bbox[3]-bbox[1]), 1.0]).reshape((4,1)))

def _associate(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    iou_matrix = _iou_batch(detections, trackers)
    x, y = linear_sum_assignment(-iou_matrix)
    matched = np.array(list(zip(x, y)))
    unmatched_dets = [d for d in range(len(detections)) if d not in matched[:,0]]
    unmatched_trks = [t for t in range(len(trackers)) if t not in matched[:,1]]
    return matched, np.array(unmatched_dets), np.array(unmatched_trks)

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0,5))):
        trks = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(self.trackers):
            trks[t] = np.concatenate([trk.predict()[0], [0]])
        matched, unmatched_dets, _ = _associate(dets, trks, self.iou_threshold)
        for m in matched: self.trackers[m[1]].update(dets[m[0], :])
        for i in unmatched_dets: self.trackers.append(_KalmanBoxTracker(dets[i, :]))
        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and trk.hit_streak >= self.min_hits:
                ret.append(np.concatenate([trk.predict()[0], [trk.id]]))
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        return np.array(ret) if ret else np.empty((0,5))
    

"""SORT (Oddiy onlayn va real vaqtda kuzatish) algoritmi 
loyihaga toʻgʻridan-toʻgʻri qoʻshilgan. U oʻzgarmas 
tezlikka asoslangan Kalman filtri orqali ishlaydi va obyektlarni 
bir-biriga moslashtirish uchun IoU (Kesishma/Birlashma nisbati) 
hamda Vengriya taqsimlash algoritmini (Hungarian algorithm) qoʻllaydi.
"""