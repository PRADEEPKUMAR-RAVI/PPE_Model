import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class SimpleTracker:
    """Simple IoU-based tracker without Kalman filter or appearance features"""

    def __init__(self, max_lost=50, iou_threshold=0.25):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.track_id_count = 0
        self.frame_count = 0

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection area
        intersect_xmin = max(x1_min, x2_min)
        intersect_ymin = max(y1_min, y2_min)
        intersect_xmax = min(x1_max, x2_max)
        intersect_ymax = min(y1_max, y2_max)

        if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
            return 0.0

        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)

        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersect_area

        if union_area == 0:
            return 0.0

        return intersect_area / union_area

    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of dicts with 'bbox', 'confidence', 'class' keys
        Returns: list of tracked objects with track_id
        """
        self.frame_count += 1

        # Handle empty detections
        if not detections:
            # Increment lost count for all tracks
            tracks_to_remove = []
            for track_id in self.tracks:
                self.tracks[track_id]['lost_count'] += 1
                if self.tracks[track_id]['lost_count'] > self.max_lost:
                    tracks_to_remove.append(track_id)

            # Remove expired tracks
            for track_id in tracks_to_remove:
                del self.tracks[track_id]

            return []

        # Extract detection data
        det_boxes = [det['bbox'] for det in detections]
        det_scores = [det['confidence'] for det in detections]
        det_classes = [det['class'] for det in detections]

        # If no existing tracks, create new ones
        if not self.tracks:
            tracked_objects = []
            for i, det in enumerate(detections):
                track_id = self._get_next_id()
                self.tracks[track_id] = {
                    'bbox': det_boxes[i],
                    'score': det_scores[i],
                    'class': det_classes[i],
                    'lost_count': 0,
                    'age': 1
                }
                tracked_objects.append({
                    'track_id': track_id,
                    'bbox': det_boxes[i],
                    'score': det_scores[i],
                    'class': det_classes[i],
                    'state': 'Tracked'
                })
            return tracked_objects

        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        iou_matrix = np.zeros((num_tracks, num_dets))

        for t_idx, track_id in enumerate(track_ids):
            track_box = self.tracks[track_id]['bbox']
            for d_idx, det_box in enumerate(det_boxes):
                iou_matrix[t_idx, d_idx] = self._calculate_iou(track_box, det_box)

        # Hungarian algorithm for optimal assignment
        matched_indices = []
        unmatched_tracks = []
        unmatched_dets = []

        if num_tracks > 0 and num_dets > 0:
            # Use Hungarian algorithm
            cost_matrix = 1 - iou_matrix
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            for t_idx, d_idx in zip(track_indices, det_indices):
                if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                    matched_indices.append((t_idx, d_idx))
                else:
                    unmatched_tracks.append(t_idx)
                    unmatched_dets.append(d_idx)

            # Find unmatched tracks and detections
            unmatched_tracks.extend([i for i in range(num_tracks) if i not in track_indices])
            unmatched_dets.extend([i for i in range(num_dets) if i not in det_indices])
        else:
            unmatched_tracks = list(range(num_tracks))
            unmatched_dets = list(range(num_dets))

        # Update matched tracks
        for t_idx, d_idx in matched_indices:
            track_id = track_ids[t_idx]
            self.tracks[track_id]['bbox'] = det_boxes[d_idx]
            self.tracks[track_id]['score'] = det_scores[d_idx]
            self.tracks[track_id]['lost_count'] = 0
            self.tracks[track_id]['age'] += 1

        # Handle unmatched tracks
        tracks_to_remove = []
        for t_idx in unmatched_tracks:
            track_id = track_ids[t_idx]
            self.tracks[track_id]['lost_count'] += 1
            if self.tracks[track_id]['lost_count'] > self.max_lost:
                tracks_to_remove.append(track_id)

        # Remove expired tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            track_id = self._get_next_id()
            self.tracks[track_id] = {
                'bbox': det_boxes[d_idx],
                'score': det_scores[d_idx],
                'class': det_classes[d_idx],
                'lost_count': 0,
                'age': 1
            }

        # Prepare output
        tracked_objects = []
        for track_id, track in self.tracks.items():
            if track['lost_count'] == 0:  # Only return visible tracks
                tracked_objects.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'score': track['score'],
                    'class': track['class'],
                    'state': 'Tracked'
                })

        return tracked_objects

    def _get_next_id(self):
        """Generate next unique track ID"""
        self.track_id_count += 1
        return self.track_id_count

    def reset(self):
        """Reset tracker state"""
        self.tracks = {}
        self.track_id_count = 0
        self.frame_count = 0