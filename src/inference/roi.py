import cv2
import numpy as np


class ROILaneCounter:
    """Counts vehicles per lane using polygon ROI regions."""

    def __init__(self, lanes: dict):
        """
        lanes: dict of {lane_name: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
        Example:
            {
                "lane_1": [[0,300],[320,300],[320,600],[0,600]],
                "lane_2": [[320,300],[640,300],[640,600],[320,600]]
            }
        """
        self.lanes = {
            name: np.array(pts, dtype=np.int32)
            for name, pts in lanes.items()
        }

    def point_in_lane(self, cx: int, cy: int, lane_name: str) -> bool:
        """Check if a point (cx, cy) is inside a lane polygon."""
        poly = self.lanes[lane_name]
        result = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)
        return result >= 0

    def count_detections(self, detections: list) -> dict:
        """
        detections: list of [x1, y1, x2, y2, class_label]
        returns: dict of {lane_name: {class_label: count}}
        """
        counts = {lane: {} for lane in self.lanes}

        for det in detections:
            x1, y1, x2, y2 = det[:4]
            label = det[4]

            # Use bounding box center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            for lane_name in self.lanes:
                if self.point_in_lane(cx, cy, lane_name):
                    counts[lane_name][label] = counts[lane_name].get(label, 0) + 1

        return counts

    def draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Draw lane polygons on a frame for visualization."""
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        for i, (lane_name, poly) in enumerate(self.lanes.items()):
            color = colors[i % len(colors)]
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
            # Label the lane
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(frame, lane_name, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame