import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def linear_assignment(cost_matrix):
    if cost_matrix.size == 0:
        return np.empty((0,2), dtype=int)
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_a = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_b = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        # ✅ FIX: Giảm noise để cải thiện độ ổn định
        self.kf.R *= 1.0  # Giảm từ 10 xuống 1 (measurement noise)
        self.kf.P[4:, 4:] *= 1.0  # Giảm từ 1000 xuống 1 (process variance)
        self.kf.P *= 1.0  # Giảm từ 10 xuống 1 (initial covariance)
        
        # ✅ Thêm process noise - smooth motion
        self.kf.Q = np.eye(7) * 0.01

        self.kf.x[:4] = bbox.reshape(-1, 1)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_bbox = bbox.copy()  # ✅ Lưu bbox trước để fallback

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_bbox = bbox.copy()  # ✅ Update bbox
        self.kf.update(bbox.reshape(-1, 1))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x[:4].reshape(-1).tolist()

class Sort:
    def __init__(self, max_age=50, min_hits=3, iou_threshold=0.5):  # ✅ Tăng max_age và iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        for det, trk in matched:
            self.trackers[trk].update(dets[det, :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((trk.kf.x[:4].reshape(-1), [trk.id])).reshape(1, -1))

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i - 1)
            i -= 1

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, dets, trks):
        if len(trks) == 0:
            return np.empty((0,2)), np.arange(len(dets)), np.empty((0,5))

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det, trk)

        matched_indices = linear_assignment(-iou_matrix)

        unmatched_dets = []
        for d in range(len(dets)):
            if d not in matched_indices[:, 0]:
                unmatched_dets.append(d)

        unmatched_trks = []
        for t in range(len(trks)):
            if t not in matched_indices[:, 1]:
                unmatched_trks.append(t)

        matches = []

        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if len(matches) == 0:
            matches = np.empty((0,2))
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_dets), np.array(unmatched_trks)