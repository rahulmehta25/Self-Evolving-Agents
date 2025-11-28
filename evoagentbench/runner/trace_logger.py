"""
Trace Logger for recording all atomic steps during agent execution.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional


class TraceLogger:
    """
    Logs all atomic events during agent execution for deterministic replay.
    """
    
    def __init__(self, trace_id: str):
        """
        Initialize the trace logger.
        
        Args:
            trace_id: Unique identifier for this trace
        """
        self.trace_id = trace_id
        self.events: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.step_index = 0
    
    def _compute_hash(self, content: Any) -> str:
        """Compute SHA256 hash of content."""
        if content is None:
            return ""
        if isinstance(content, (dict, list)):
            json_str = json.dumps(content, sort_keys=True)
            return hashlib.sha256(json_str.encode()).hexdigest()
        return hashlib.sha256(str(content).encode()).hexdigest()
    
    def log_event(self, event_type: str, payload: Dict[str, Any],
                  input_content: Optional[Any] = None,
                  output_content: Optional[Any] = None):
        """
        Log an event to the trace.
        
        Args:
            event_type: Type of event (MODEL_INPUT, MODEL_OUTPUT, TOOL_CALL, etc.)
            payload: Event-specific data
            input_content: Input content for hashing (e.g., prompt text)
            output_content: Output content for hashing (e.g., LLM response)
        """
        timestamp = time.time() - self.start_time
        input_hash = self._compute_hash(input_content) if input_content is not None else None
        output_hash = self._compute_hash(output_content) if output_content is not None else None
        
        event = {
            "trace_id": self.trace_id,
            "step_index": self.step_index,
            "timestamp": timestamp,
            "event_type": event_type,
            "payload": payload,
            "input_hash": input_hash,
            "output_hash": output_hash
        }
        
        self.events.append(event)
        self.step_index += 1
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get the complete trace."""
        return self.events.copy()
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all events of a specific type."""
        return [event for event in self.events if event["event_type"] == event_type]
