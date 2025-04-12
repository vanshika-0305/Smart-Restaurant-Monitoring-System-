import time
from datetime import datetime

class TableStateTracker:
    """
    Tracks the states of restaurant tables over time and generates alerts
    when tables remain in 'unoccupied_dirty' state for too long.
    """
    
    def __init__(self, table_ids, dirty_threshold_seconds=300):
        """
        Initialize the table tracker.
        
        Args:
            table_ids (list): List of table IDs to track
            dirty_threshold_seconds (int): Time threshold in seconds for dirty table alerts
        """
        self.dirty_threshold = dirty_threshold_seconds
        self.table_states = {}
        
        # Initialize tables with clean state
        for table_id in table_ids:
            self.table_states[table_id] = {
                "state": "unoccupied_clean",
                "last_state_change": time.time(),
                "last_dirty_time": None,
                "dirty_duration": 0
            }
        
        self.alerts = []
    
    def update_table_state(self, table_id, new_state):
        """
        Update the state of a specific table and manage state transitions.
        
        Args:
            table_id (str): The ID of the table (e.g., "T1")
            new_state (str): New state - "unoccupied_clean", "unoccupied_dirty", or "occupied"
        
        Returns:
            bool: True if the state changed, False otherwise
        """
        if table_id not in self.table_states:
            raise ValueError(f"Table ID {table_id} not found in tracked tables")
        
        current_time = time.time()
        current_state = self.table_states[table_id]["state"]
        
        # If state hasn't changed, update dirty duration if applicable
        if current_state == new_state:
            if new_state == "unoccupied_dirty":
                self.table_states[table_id]["dirty_duration"] = (
                    current_time - self.table_states[table_id]["last_dirty_time"]
                )
            return False
        
        # Handle state transition
        self.table_states[table_id]["state"] = new_state
        self.table_states[table_id]["last_state_change"] = current_time
        
        # If newly dirty, set last_dirty_time
        if new_state == "unoccupied_dirty":
            self.table_states[table_id]["last_dirty_time"] = current_time
            self.table_states[table_id]["dirty_duration"] = 0
        
        # If no longer dirty, reset dirty tracking
        elif current_state == "unoccupied_dirty":
            self.table_states[table_id]["dirty_duration"] = 0
            self.table_states[table_id]["last_dirty_time"] = None
        
        return True
    
    def check_alerts(self):
        """
        Check for tables that have been dirty for too long and generate alerts.
        
        Returns:
            list: List of alert dictionaries for tables exceeding the dirty threshold
        """
        current_time = time.time()
        self.alerts = []
        
        for table_id, state_info in self.table_states.items():
            if state_info["state"] == "unoccupied_dirty" and state_info["last_dirty_time"]:
                dirty_duration = current_time - state_info["last_dirty_time"]
                self.table_states[table_id]["dirty_duration"] = dirty_duration
                
                if dirty_duration >= self.dirty_threshold:
                    time_str = str(datetime.now().strftime("%H:%M:%S"))
                    alert = {
                        "table_id": table_id,
                        "duration": int(dirty_duration),
                        "timestamp": time_str,
                        "message": f"Table {table_id} has been dirty for {int(dirty_duration)}s"
                    }
                    self.alerts.append(alert)
        
        return self.alerts
    
    def get_table_state(self, table_id):
        """
        Get the current state information for a specific table.
        
        Args:
            table_id (str): The ID of the table
            
        Returns:
            dict: Table state information
        """
        if table_id not in self.table_states:
            raise ValueError(f"Table ID {table_id} not found in tracked tables")
        
        return self.table_states[table_id]
    
    def get_all_states(self):
        """
        Get states of all tables.
        
        Returns:
            dict: Dictionary with all table states
        """
        return self.table_states
