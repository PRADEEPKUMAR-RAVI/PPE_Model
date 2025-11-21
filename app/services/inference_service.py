



import cv2
import os
import tempfile
import uuid
from ultralytics import YOLO
from app.services.tracking_service import TrackingService


class InferenceService:

    def __init__(self):
        self.allowed_video_types = [
            "video/mp4",
            "video/avi",
            "video/mov",
            "video/mkv",
            "video/webm",
            "video/quicktime"
        ]
        self.model = YOLO("/home/softsuave/Pictures/PPE_model/Large model/best.pt")
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.tracking_service = TrackingService()
        self.frame_counter = 0

    def detect_frame(self, frame):
        """Extract raw detections from a single frame and group PPE with humans"""
        self.frame_counter += 1
        results = self.model(frame, conf=0.25, imgsz=1280)
        print(f"Processing frame: {self.frame_counter}")

        # Define allowed classes
        allowed_classes = ["Helmet", "Vest"]

        # Separate humans and PPE items
        humans = []
        ppe_items = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    if class_name not in allowed_classes:
                        continue

                    detection = {
                        "class": class_name,
                        "confidence": float(confidence),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    }

                    if class_name == 'Human':
                        humans.append(detection)
                    else:
                        ppe_items.append(detection)

        # For each human, find associated PPE items
        frame_detections = []
        for human in humans:
            human_bbox = human['bbox']

            # Check which PPE items are associated with this human
            ppe_status = {
                'helmet': False,
                'vest': False,
                'shoe': False,
                'gloves': False
            }

            for ppe in ppe_items:
                if self._is_ppe_associated_with_human(ppe['bbox'], human_bbox):
                    ppe_class = ppe['class'].lower()
                    if ppe_class == 'helmet':
                        ppe_status['helmet'] = True
                    elif ppe_class == 'vest':
                        ppe_status['vest'] = True
                    elif ppe_class == 'shoe':
                        ppe_status['shoe'] = True
                    elif ppe_class in ['gloves']:
                        ppe_status['gloves'] = True

            # Check if all PPE is present
            all_ppe_present = all(ppe_status.values())

            frame_detections.append({
                "class": "Human",
                "confidence": human['confidence'],
                "bbox": human_bbox,
                "ppe_status": ppe_status,
                "all_ppe_present": all_ppe_present
            })

        return frame_detections

    def _is_ppe_associated_with_human(self, ppe_bbox, human_bbox):
        """Check if PPE item is associated with a human using IoU and containment"""
        # Calculate IoU (Intersection over Union)
        x1_inter = max(ppe_bbox[0], human_bbox[0])
        y1_inter = max(ppe_bbox[1], human_bbox[1])
        x2_inter = min(ppe_bbox[2], human_bbox[2])
        y2_inter = min(ppe_bbox[3], human_bbox[3])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return False

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        ppe_area = (ppe_bbox[2] - ppe_bbox[0]) * (ppe_bbox[3] - ppe_bbox[1])
        human_area = (human_bbox[2] - human_bbox[0]) * (human_bbox[3] - human_bbox[1])

        # Check if PPE overlaps significantly with human
        # Use IoU or check if PPE center is within human bbox
        iou = inter_area / (ppe_area + human_area - inter_area)

        # Also check if PPE center is within human bbox (with some margin)
        ppe_center_x = (ppe_bbox[0] + ppe_bbox[2]) / 2
        ppe_center_y = (ppe_bbox[1] + ppe_bbox[3]) / 2

        # Expand human bbox by 10% to account for slight misalignments
        margin = 0.1
        h_width = human_bbox[2] - human_bbox[0]
        h_height = human_bbox[3] - human_bbox[1]
        expanded_human = [
            human_bbox[0] - margin * h_width,
            human_bbox[1] - margin * h_height,
            human_bbox[2] + margin * h_width,
            human_bbox[3] + margin * h_height
        ]

        center_in_human = (expanded_human[0] <= ppe_center_x <= expanded_human[2] and
                          expanded_human[1] <= ppe_center_y <= expanded_human[3])

        # Return true if IoU is significant OR center is within human bbox
        return iou > 0.1 or center_in_human

    async def process_file(self, file):
        if file.content_type not in self.allowed_video_types:
            raise ValueError(f"File type '{file.content_type}' is not supported. Only video files are allowed.")

        temp_input_path = None
        output_filename = None

        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_input_path = temp_file.name

            # Generate unique output filename
            unique_id = str(uuid.uuid4())
            output_filename = f"processed_{unique_id}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)

            # Reset frame counter and tracker for new video
            self.frame_counter = 0
            self.tracking_service.reset()

            # Open input video
            cap = cv2.VideoCapture(temp_input_path)

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total frames (fast method): {total_frames}")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                print(frame_count)
                # Get raw detections from YOLO
                raw_detections = self.detect_frame(frame)

                # Update tracking with current detections
                tracked_objects = self.tracking_service.update_tracks(raw_detections)

                # Draw tracking information on frame
                frame = self.tracking_service.draw_tracks(frame, tracked_objects)

                # Write annotated frame with tracking info
                out.write(frame)
                frame_count += 1

            # Release resources
            cap.release()
            out.release()

            # Get tracking analytics
            analytics = self.tracking_service.get_analytics()
            analytics["frames_processed"] = frame_count
            analytics["output_video"] = output_filename

            return analytics

        finally:
            # Cleanup temporary file
            if temp_input_path and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)