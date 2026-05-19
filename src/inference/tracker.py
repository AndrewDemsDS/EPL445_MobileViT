import numpy as np
from filterpy.kalman import KalmanFilter


class SORTTracker:
    """Lightweight SORT tracker for vehicle bounding boxes."""

    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self._next_id = 1

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, class_label]
        returns:    list of [x1, y1, x2, y2, track_id, class_label]
        """
        self.frame_count += 1
        boxes = np.array([d[:4] for d in detections]) if detections else np.empty((0, 4))
        labels = [d[4] for d in detections] if detections else []

        # Predict existing trackers
        for t in self.trackers:
            t["box"] = t["box"]  # simple passthrough (no motion model for now)
            t["age"] += 1
            t["hits_since_update"] += 1

        # Match detections to trackers via IoU
        matched, unmatched_dets = self._match(boxes)

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx]["box"] = boxes[det_idx]
            self.trackers[trk_idx]["label"] = labels[det_idx]
            self.trackers[trk_idx]["hits"] += 1
            self.trackers[trk_idx]["hits_since_update"] = 0

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            self.trackers.append({
                "id": self._next_id,
                "box": boxes[det_idx],
                "label": labels[det_idx],
                "age": 1,
                "hits": 1,
                "hits_since_update": 0,
            })
            self._next_id += 1

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t["hits_since_update"] <= self.max_age]

        # Return confirmed trackers
        results = []
        for t in self.trackers:
            if t["hits"] >= self.min_hits:
                x1, y1, x2, y2 = t["box"].astype(int)
                results.append([x1, y1, x2, y2, t["id"], t["label"]])
        return results

    def _match(self, boxes):
        if len(self.trackers) == 0 or len(boxes) == 0:
            return [], list(range(len(boxes)))

        trk_boxes = np.array([t["box"] for t in self.trackers])
        iou_matrix = self._iou_batch(boxes, trk_boxes)

        matched = []
        unmatched_dets = []
        used_trks = set()

        for det_idx in range(len(boxes)):
            best_trk = int(np.argmax(iou_matrix[det_idx]))
            if iou_matrix[det_idx][best_trk] >= self.iou_threshold and best_trk not in used_trks:
                matched.append((det_idx, best_trk))
                used_trks.add(best_trk)
            else:
                unmatched_dets.append(det_idx)

        return matched, unmatched_dets

    @staticmethod
    def _iou_batch(boxes_a, boxes_b):
        iou = np.zeros((len(boxes_a), len(boxes_b)))
        for i, a in enumerate(boxes_a):
            for j, b in enumerate(boxes_b):
                xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
                xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                union = area_a + area_b - inter
                iou[i][j] = inter / union if union > 0 else 0
        return iou