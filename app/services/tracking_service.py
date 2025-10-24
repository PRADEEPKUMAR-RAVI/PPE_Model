import cv2
from collections import defaultdict
from app.services.simple_tracker import SimpleTracker


class TrackingService:
    def __init__(self):
        # Use SimpleTracker instead of ByteTracker
        self.tracker = SimpleTracker(max_lost=50, iou_threshold=0.25)
        self.track_history = defaultdict(list)
        self.track_stats = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'total_frames': 0,
            'class_name': None,
            'avg_confidence': 0,
            'confidence_sum': 0
        })
        self.frame_id = 0

        # Define class-specific colors (BGR format)
        self.class_colors = {
            'Helmet': (255, 0, 0),      # Blue
            'Vest': (0, 255, 0),        # Green
            'Shoe': (0, 0, 255),        # Red
            'Gloves': (0, 255, 255),    # Yellow
            'gloves': (0, 255, 255),    # Yellow
            'Human': (255, 0, 255)      # Magenta
        }

    def update_tracks(self, frame, raw_detections):
        """Update tracks with new detections"""
        if not raw_detections:
            # Still update tracker with empty detections to handle lost tracks
            self.tracker.update([])
            self.frame_id += 1
            return []

        self.frame_id += 1

        # Update tracker with detections
        tracked_objects = self.tracker.update(raw_detections)

        # Update tracking history and statistics
        for track in tracked_objects:
            track_id = track['track_id']

            # Store track history for visualization
            self.track_history[track_id].append({
                'frame': self.frame_id,
                'bbox': track['bbox'],
                'confidence': track['score']
            })

            # Keep only last 30 frames of history for trail visualization
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # Update statistics
            stats = self.track_stats[track_id]
            if stats['first_seen'] is None:
                stats['first_seen'] = self.frame_id
                stats['class_name'] = track['class']

            stats['last_seen'] = self.frame_id
            stats['total_frames'] += 1
            stats['confidence_sum'] += track['score']
            stats['avg_confidence'] = stats['confidence_sum'] / stats['total_frames']

        return tracked_objects

    def draw_tracks(self, frame, tracked_objects):
        """Draw bounding boxes and trails on frame"""
        for track in tracked_objects:
            track_id = track['track_id']
            bbox = track['bbox']
            confidence = track['score']
            class_name = track['class']

            # Convert to integers for drawing
            x1, y1, x2, y2 = map(int, bbox)
            # Use class-based color, fallback to white if class not in mapping
            color = self.class_colors.get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with ID, class, and confidence
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Background for text
            cv2.rectangle(frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)

            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw trail
            history = self.track_history[track_id]
            if len(history) > 1:
                points = []
                for h in history[-10:]:  # Last 10 points for trail
                    bbox_h = h['bbox']
                    center = (int((bbox_h[0] + bbox_h[2]) / 2),
                             int((bbox_h[1] + bbox_h[3]) / 2))
                    points.append(center)

                # Draw trail with fading effect
                for i in range(1, len(points)):
                    thickness = max(1, int(3 * (i / len(points))))  # Ensure thickness is at least 1
                    cv2.line(frame, points[i-1], points[i], color, thickness)

        return frame

    def get_analytics(self):
        """Get tracking analytics"""
        unique_objects = {}
        total_duration = 0

        for track_id, stats in self.track_stats.items():
            class_name = stats['class_name']
            duration = stats['last_seen'] - stats['first_seen'] + 1 if stats['last_seen'] else 1

            if class_name not in unique_objects:
                unique_objects[class_name] = {
                    'count': 0,
                    'total_duration': 0,
                    'avg_confidence': 0,
                    'confidence_sum': 0
                }

            unique_objects[class_name]['count'] += 1
            unique_objects[class_name]['total_duration'] += duration
            unique_objects[class_name]['confidence_sum'] += stats['confidence_sum']
            total_duration += duration

        # Calculate averages
        for class_name, data in unique_objects.items():
            if data['total_duration'] > 0:
                data['avg_duration'] = data['total_duration'] / data['count']
                data['avg_confidence'] = data['confidence_sum'] / data['total_duration']

        # Calculate overall tracking confidence
        tracking_confidence = 0
        if self.track_stats:
            tracking_confidence = sum(stats['avg_confidence'] for stats in self.track_stats.values()) / len(self.track_stats)

        # Count currently active tracks
        active_tracks = 0
        for track_id in self.tracker.tracks:
            if self.tracker.tracks[track_id]['lost_count'] == 0:
                active_tracks += 1

        return {
            'unique_objects_detected': sum(data['count'] for data in unique_objects.values()),
            'object_tracks': unique_objects,
            'tracking_confidence': round(tracking_confidence, 3),
            'total_unique_tracks': len(self.track_stats),
            'active_tracks': active_tracks,
            'frames_processed': self.frame_id
        }

    def reset(self):
        """Reset tracker for new video"""
        self.tracker.reset()
        self.track_history.clear()
        self.track_stats.clear()
        self.frame_id = 0