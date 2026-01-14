"""
FER Status Tracker

Manages in-memory storage for FER processing results and requests.
Provides thread-safe read/write operations for status monitoring.
"""

import logging
import threading
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# In-memory storage for FER results and requests
_recent_requests = deque(maxlen=100)  # Store up to 100 recent requests
_recent_results = deque(maxlen=100)   # Store up to 100 recent results

# Thread locks for safe concurrent access
_requests_lock = threading.Lock()
_results_lock = threading.Lock()


def log_request(
    user_id: str,
    timestamp: str,
    filename: Optional[str] = None
) -> None:
    """
    Log an FER request to in-memory storage.

    Args:
        user_id: User identifier
        timestamp: Timestamp when request was received (ISO format)
        filename: Optional filename of the uploaded image
    """
    try:
        log_entry = {
            "user_id": user_id,
            "timestamp": timestamp,
            "filename": filename or "image.jpg",
            "status": "received"
        }

        with _requests_lock:
            _recent_requests.append(log_entry)

        logger.debug(f"Logged FER request for user {user_id}")

    except Exception as e:
        logger.error(f"Error logging FER request: {e}", exc_info=True)


def log_result(
    user_id: str,
    timestamp: str,
    emotion: str,
    confidence: float,
    db_write_success: bool = False,
    filename: Optional[str] = None
) -> None:
    """
    Log an FER processing result to in-memory storage.

    Args:
        user_id: User identifier
        timestamp: Timestamp when request was received (ISO format)
        emotion: Detected emotion
        confidence: Confidence score
        db_write_success: Whether database write succeeded
        filename: Optional filename of the processed image
    """
    try:
        log_entry = {
            "user_id": user_id,
            "timestamp": timestamp,
            "filename": filename or "image.jpg",
            "emotion": emotion,
            "emotion_confidence": confidence,
            "db_write_success": db_write_success,
            "status": "completed"
        }

        with _results_lock:
            _recent_results.append(log_entry)

        logger.debug(f"Logged FER result for user {user_id}: {emotion} ({confidence:.2f})")

    except Exception as e:
        logger.error(f"Error logging FER result: {e}", exc_info=True)


def read_recent_requests(
    limit: int = 20,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Read recent requests from in-memory storage.

    Args:
        limit: Maximum number of entries to return
        user_id: Optional filter by user_id

    Returns:
        List of request dictionaries (newest first)
    """
    try:
        with _requests_lock:
            # Convert deque to list (deque maintains insertion order, newest last)
            # We want newest first, so reverse it
            all_entries = list(_recent_requests)
            all_entries.reverse()  # Newest first

        # Filter by user_id if provided
        if user_id:
            all_entries = [entry for entry in all_entries if entry.get("user_id") == user_id]

        # Return last N entries (already newest first)
        return all_entries[:limit]

    except Exception as e:
        logger.error(f"Error reading recent requests: {e}", exc_info=True)
        return []


def read_recent_results(
    limit: int = 20,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Read recent results from in-memory storage.

    Args:
        limit: Maximum number of entries to return
        user_id: Optional filter by user_id

    Returns:
        List of result dictionaries (newest first)
    """
    try:
        with _results_lock:
            # Convert deque to list (deque maintains insertion order, newest last)
            # We want newest first, so reverse it
            all_entries = list(_recent_results)
            all_entries.reverse()  # Newest first

        # Filter by user_id if provided
        if user_id:
            all_entries = [entry for entry in all_entries if entry.get("user_id") == user_id]

        # Return last N entries (already newest first)
        return all_entries[:limit]

    except Exception as e:
        logger.error(f"Error reading recent results: {e}", exc_info=True)
        return []


def get_request_count() -> int:
    """Get the current number of requests in memory."""
    with _requests_lock:
        return len(_recent_requests)


def get_result_count() -> int:
    """Get the current number of results in memory."""
    with _results_lock:
        return len(_recent_results)


def clear_requests() -> None:
    """Clear all requests from memory."""
    with _requests_lock:
        _recent_requests.clear()
    logger.info("Cleared all FER requests")


def clear_results() -> None:
    """Clear all results from memory."""
    with _results_lock:
        _recent_results.clear()
    logger.info("Cleared all FER results")
