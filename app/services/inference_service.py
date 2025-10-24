



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
        self.model = YOLO("/home/softsuave/Pictures/PPE_model/pure-model with 150 Epochs/best.pt")
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.tracking_service = TrackingService()
        self.frame_counter = 0

    def detect_frame(self, frame):
        """Extract raw detections from a single frame"""
        self.frame_counter += 1
        results = self.model(frame, conf=0.25, iou=0.25, imgsz=1280)
        print(f"Processing frame: {self.frame_counter}")
        frame_detections = []

        # Define allowed classes
        allowed_classes = ['Gloves', 'Helmet', 'Human', 'Shoe', 'Vest', 'gloves']

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    # Only add detections for helmet and vest
                    if class_name in allowed_classes:
                        frame_detections.append({
                            "class": class_name,
                            "confidence": float(confidence),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })

        return frame_detections

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
                tracked_objects = self.tracking_service.update_tracks(frame, raw_detections)

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