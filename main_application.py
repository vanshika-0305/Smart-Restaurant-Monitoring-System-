import cv2
import json
import time
import torch
import numpy as np
from ultralytics import YOLO
from table_state_tracker import TableStateTracker
from visualization import RestaurantVisualizer

class RestaurantMonitor:
    """
    Main application for the Smart Restaurant Table Monitoring System.
    Integrates YOLOv8 detection with table tracking and visualization.
    """
    
    def __init__(self, model_path, config_path, source=0, dirty_threshold=300):
        """
        Initialize the restaurant monitoring system.
        
        Args:
            model_path (str): Path to the YOLOv8 model file
            config_path (str): Path to the table layout configuration JSON
            source (int or str): Camera index or video file path
            dirty_threshold (int): Time in seconds before dirty table alert
        """
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # Load table configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set up video source
        self.source = source
        
        # Extract table IDs from config
        self.table_ids = list(self.config["tables"].keys())
        
        # Initialize table tracker
        self.tracker = TableStateTracker(self.table_ids, dirty_threshold)
        
        # Initialize visualizer
        self.visualizer = RestaurantVisualizer()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
    def _map_detection_to_table(self, detection_boxes):
        """
        Map detected objects to predefined table positions.
        
        Args:
            detection_boxes (list): List of detection boxes with class
            
        Returns:
            dict: Mapping of table_id to detected state
        """
        table_states = {}
        
        # For each predefined table in our configuration
        for table_id, table_box in self.config["tables"].items():
            tx1, ty1, tx2, ty2 = table_box
            table_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
            
            # Default state if no detection matches
            best_state = "unoccupied_clean"
            best_iou = 0.3  # Minimum IoU threshold for a match
            
            # Check each detection for overlap with this table
            for det_box, det_class in detection_boxes:
                dx1, dy1, dx2, dy2 = det_box
                
                # Calculate IoU between the predefined table box and detection box
                x_left = max(tx1, dx1)
                y_top = max(ty1, dy1)
                x_right = min(tx2, dx2)
                y_bottom = min(ty2, dy2)
                
                if x_right < x_left or y_bottom < y_top:
                    continue  # No overlap
                
                intersection = (x_right - x_left) * (y_bottom - y_top)
                table_area = (tx2 - tx1) * (ty2 - ty1)
                detection_area = (dx2 - dx1) * (dy2 - dy1)
                union = table_area + detection_area - intersection
                iou = intersection / union
                
                if iou > best_iou:
                    best_iou = iou
                    best_state = self.class_names[det_class]
            
            table_states[table_id] = best_state
        
        return table_states
    
    def process_frame(self, frame):
        """
        Process a single video frame.
        
        Args:
            frame (np.array): Input video frame
            
        Returns:
            np.array: Processed frame with visualizations
        """
        self.frame_count += 1
        
        # Resize frame if needed
        height, width = frame.shape[:2]
        
        # Make sure table coordinates work with actual frame size
        # This is important if the config was created for a different resolution
        scale_x = width / self.config["reference_width"]
        scale_y = height / self.config["reference_height"]
        
        # Run YOLOv8 detection
        results = self.model(frame)
        
        # Extract detection boxes and classes
        detection_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                detection_boxes.append(([x1, y1, x2, y2], cls))
        
        # Map detections to predefined table locations
        table_detection_states = self._map_detection_to_table(detection_boxes)
        
        # Update table states in tracker
        for table_id, state in table_detection_states.items():
            self.tracker.update_table_state(table_id, state)
        
        # Check for alerts
        alerts = self.tracker.check_alerts()
        
        # Get all table states for visualization
        table_states = self.tracker.get_all_states()
        
        # Draw table boxes and states
        for table_id, box in self.config["tables"].items():
            # Scale boxes to match current frame size
            scaled_box = [
                int(box[0] * scale_x),
                int(box[1] * scale_y),
                int(box[2] * scale_x),
                int(box[3] * scale_y)
            ]
            
            # Get current state and dirty duration if applicable
            state_info = table_states[table_id]
            state = state_info["state"]
            dirty_duration = state_info.get("dirty_duration", None)
            
            # Draw on frame
            frame = self.visualizer.draw_table_box(
                frame, 
                table_id, 
                scaled_box, 
                state, 
                dirty_duration
            )
        
        # Add alert banner if needed
        if alerts:
            frame = self.visualizer.draw_alerts_banner(frame, alerts)
        
        # Add dashboard with stats
        frame = self.visualizer.draw_dashboard_stats(frame, table_states)
        
        # Calculate FPS
        if self.frame_count % 30 == 0:
            end_time = time.time()
            self.fps = 30 / (end_time - self.start_time)
            self.start_time = end_time
        
        # Add FPS counter
        cv2.putText(
            frame, 
            f"FPS: {self.fps:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return frame

    def run(self):
        """
        Main loop to process video stream.
        """
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source {self.source}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream")
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the result
            cv2.imshow("Restaurant Table Monitor", processed_frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example configuration
    model_path = "model/trained_model.pt"  # YOLOv8 model file
    config_path = "configs/table_layout.json"  # Table layout configuration
    video_source = "video2.mp4"  # Demo video file or webcam (0)
    
    # Create and run the monitoring system
    monitor = RestaurantMonitor(model_path, config_path, video_source)
    monitor.run()
