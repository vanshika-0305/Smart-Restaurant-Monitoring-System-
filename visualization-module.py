import cv2
import numpy as np
from datetime import datetime

class RestaurantVisualizer:
    """
    Handles visualization of table states in the restaurant monitoring system.
    Draws bounding boxes, labels, and alerts on the video feed.
    """
    
    def __init__(self):
        """Initialize the visualizer with colors for different table states."""
        self.colors = {
            "unoccupied_clean": (0, 255, 0),    # Green
            "unoccupied_dirty": (0, 0, 255),    # Red
            "occupied": (255, 0, 0)             # Blue
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_thickness = 2
        self.text_size = 0.6
        self.text_thickness = 2
    
    def draw_table_box(self, frame, table_id, box, state, dirty_duration=None):
        """
        Draw a colored bounding box and label for a table.
        
        Args:
            frame (np.array): The image frame to draw on
            table_id (str): Table identifier (e.g., "T1")
            box (tuple): Bounding box coordinates (x1, y1, x2, y2)
            state (str): Current table state
            dirty_duration (float, optional): Duration table has been dirty in seconds
        
        Returns:
            np.array: Frame with the table visualization added
        """
        x1, y1, x2, y2 = box
        color = self.colors.get(state, (128, 128, 128))  # Default gray if unknown state
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
        
        # Prepare label text
        label_text = f"{table_id}: {state}"
        
        # Add timer if dirty
        if state == "unoccupied_dirty" and dirty_duration is not None:
            mins, secs = divmod(int(dirty_duration), 60)
            label_text += f" [{mins:02d}:{secs:02d}]"
        
        # Draw label background
        text_size = cv2.getTextSize(label_text, self.font, self.text_size, self.text_thickness)[0]
        cv2.rectangle(
            frame, 
            (x1, y1 - 25), 
            (x1 + text_size[0] + 10, y1), 
            color, 
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            frame, 
            label_text, 
            (x1 + 5, y1 - 5), 
            self.font, 
            self.text_size, 
            (255, 255, 255),  # White text
            self.text_thickness
        )
        
        return frame
    
    def draw_alerts_banner(self, frame, alerts):
        """
        Draw an alert banner at the top of the frame if there are any alerts.
        
        Args:
            frame (np.array): The image frame to draw on
            alerts (list): List of alert dictionaries
        
        Returns:
            np.array: Frame with alert banner added
        """
        if not alerts:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        banner_height = 40 * len(alerts) + 10  # Height depends on number of alerts
        
        # Create alert banner
        banner = np.zeros((banner_height, frame_width, 3), dtype=np.uint8)
        banner[:, :] = (0, 0, 150)  # Dark red background
        
        # Add current time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            banner, 
            f"ALERTS ({time_str}):", 
            (10, 30), 
            self.font, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Add alert messages
        for i, alert in enumerate(alerts):
            y_pos = 30 * (i + 1) + 10
            cv2.putText(
                banner, 
                alert["message"], 
                (20, y_pos), 
                self.font, 
                0.6, 
                (255, 255, 255), 
                1
            )
        
        # Combine banner with frame
        combined = np.vstack((banner, frame))
        
        return combined
    
    def draw_dashboard_stats(self, frame, table_states):
        """
        Draw a stats dashboard on the right side of the frame.
        
        Args:
            frame (np.array): The image frame to draw on
            table_states (dict): Dictionary of all table states
        
        Returns:
            np.array: Frame with dashboard added
        """
        frame_height, frame_width = frame.shape[:2]
        dashboard_width = 200
        
        # Create dashboard area
        dashboard = np.ones((frame_height, dashboard_width, 3), dtype=np.uint8) * 240  # Light gray
        
        # Calculate statistics
        stats = {
            "unoccupied_clean": 0,
            "unoccupied_dirty": 0,
            "occupied": 0
        }
        
        for table_id, info in table_states.items():
            state = info["state"]
            stats[state] += 1
        
        # Draw header
        cv2.putText(
            dashboard, 
            "TABLE STATS", 
            (10, 30), 
            self.font, 
            0.7, 
            (0, 0, 0), 
            2
        )
        
        # Draw statistics
        y_pos = 70
        for state, count in stats.items():
            color = self.colors.get(state, (128, 128, 128))
            cv2.rectangle(dashboard, (10, y_pos-15), (30, y_pos+5), color, -1)
            cv2.putText(
                dashboard, 
                f"{state}: {count}", 
                (40, y_pos), 
                self.font, 
                0.6, 
                (0, 0, 0), 
                1
            )
            y_pos += 40
        
        # Table list
        cv2.putText(
            dashboard, 
            "TABLE STATUS:", 
            (10, y_pos + 20), 
            self.font, 
            0.7, 
            (0, 0, 0), 
            2
        )
        
        y_pos += 60
        for table_id, info in table_states.items():
            state = info["state"]
            color = self.colors.get(state, (128, 128, 128))
            
            # Table indicator
            cv2.rectangle(dashboard, (10, y_pos-15), (30, y_pos+5), color, -1)
            
            # Table text
            status_text = f"{table_id}: {state}"
            cv2.putText(
                dashboard, 
                status_text, 
                (40, y_pos), 
                self.font, 
                0.6, 
                (0, 0, 0), 
                1
            )
            
            y_pos += 30
        
        # Combine frame with dashboard
        combined = np.hstack((frame, dashboard))
        
        return combined
