"""
Event-Driven Communication System

Agent D: State Management & Integration Specialist
Event bus for state change notifications and system-wide communication.
"""

import asyncio
import threading
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class StateChangeEvent:
    """Event representing a state change"""
    change_id: str
    entity_id: str
    change_type: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSubscription:
    """Event subscription information"""
    event_type: str
    callback: Callable
    subscriber_id: str
    created_timestamp: datetime = field(default_factory=datetime.now)
    active: bool = True


class EventBus:
    """
    Event bus for system-wide communication
    
    Provides pub/sub messaging for state changes and system events.
    Thread-safe and supports both sync and async subscribers.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._event_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._max_history = 1000
        
        # Performance metrics
        self.metrics = {
            "events_published": 0,
            "events_delivered": 0,
            "failed_deliveries": 0,
            "active_subscriptions": 0
        }
    
    def subscribe(self, event_type: str, callback: Callable, 
                 subscriber_id: str = None) -> str:
        """Subscribe to events of a specific type"""
        with self._lock:
            if subscriber_id is None:
                subscriber_id = f"subscriber_{len(self._subscribers[event_type])}"
            
            subscription = EventSubscription(
                event_type=event_type,
                callback=callback,
                subscriber_id=subscriber_id
            )
            
            self._subscribers[event_type].append(subscription)
            self.metrics["active_subscriptions"] += 1
            
            logger.debug(f"Subscribed {subscriber_id} to {event_type}")
            return subscriber_id
    
    def unsubscribe(self, event_type: str, callback: Callable = None, 
                   subscriber_id: str = None) -> bool:
        """Unsubscribe from events"""
        with self._lock:
            if event_type not in self._subscribers:
                return False
            
            subscriptions = self._subscribers[event_type]
            
            # Find subscription to remove
            to_remove = []
            for subscription in subscriptions:
                if callback and subscription.callback == callback:
                    to_remove.append(subscription)
                elif subscriber_id and subscription.subscriber_id == subscriber_id:
                    to_remove.append(subscription)
            
            # Remove subscriptions
            for subscription in to_remove:
                subscriptions.remove(subscription)
                self.metrics["active_subscriptions"] -= 1
                logger.debug(f"Unsubscribed {subscription.subscriber_id} from {event_type}")
            
            return len(to_remove) > 0
    
    def publish(self, event_type: str, event_data: Any):
        """Publish event to all subscribers"""
        with self._lock:
            # Record event
            event_record = {
                "event_type": event_type,
                "timestamp": datetime.now(),
                "data": event_data,
                "subscribers_notified": 0,
                "delivery_failures": 0
            }
            
            self.metrics["events_published"] += 1
            
            # Notify subscribers
            if event_type in self._subscribers:
                active_subscribers = [s for s in self._subscribers[event_type] if s.active]
                
                for subscription in active_subscribers:
                    try:
                        subscription.callback(event_data)
                        event_record["subscribers_notified"] += 1
                        self.metrics["events_delivered"] += 1
                    except Exception as e:
                        event_record["delivery_failures"] += 1
                        self.metrics["failed_deliveries"] += 1
                        logger.error(f"Failed to deliver event {event_type} to "
                                   f"{subscription.subscriber_id}: {e}")
            
            # Store in history
            self._event_history.append(event_record)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            
            logger.debug(f"Published {event_type} to {event_record['subscribers_notified']} subscribers")
    
    def get_subscribers(self, event_type: str) -> List[EventSubscription]:
        """Get list of subscribers for event type"""
        with self._lock:
            return [s for s in self._subscribers.get(event_type, []) if s.active]
    
    def get_event_history(self, event_type: str = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event history"""
        with self._lock:
            history = self._event_history
            
            if event_type:
                history = [e for e in history if e["event_type"] == event_type]
            
            return history[-limit:] if limit else history
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus performance metrics"""
        with self._lock:
            return {
                **self.metrics.copy(),
                "event_types": list(self._subscribers.keys()),
                "total_subscribers": sum(len(subs) for subs in self._subscribers.values()),
                "event_history_size": len(self._event_history)
            }
    
    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
    
    def shutdown(self):
        """Shutdown event bus"""
        with self._lock:
            self._subscribers.clear()
            self._event_history.clear()
            self.metrics = {k: 0 for k in self.metrics}
            logger.info("Event bus shutdown complete")


class AsyncEventBus(EventBus):
    """Async version of event bus"""
    
    def __init__(self):
        super().__init__()
        self._async_lock = asyncio.Lock()
    
    async def subscribe_async(self, event_type: str, callback: Callable,
                            subscriber_id: str = None) -> str:
        """Async version of subscribe"""
        async with self._async_lock:
            return self.subscribe(event_type, callback, subscriber_id)
    
    async def publish_async(self, event_type: str, event_data: Any):
        """Async version of publish"""
        async with self._async_lock:
            # Handle async callbacks
            if event_type in self._subscribers:
                active_subscribers = [s for s in self._subscribers[event_type] if s.active]
                
                async_tasks = []
                for subscription in active_subscribers:
                    if asyncio.iscoroutinefunction(subscription.callback):
                        task = subscription.callback(event_data)
                        async_tasks.append(task)
                    else:
                        # Handle sync callbacks in async context
                        try:
                            subscription.callback(event_data)
                        except Exception as e:
                            logger.error(f"Sync callback failed: {e}")
                
                # Wait for all async callbacks
                if async_tasks:
                    await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Use parent publish for recording
            self.publish(event_type, event_data)


class EventFilter:
    """Filter events based on criteria"""
    
    def __init__(self, criteria: Dict[str, Any] = None):
        self.criteria = criteria or {}
    
    def matches(self, event_type: str, event_data: Any) -> bool:
        """Check if event matches filter criteria"""
        if "event_type" in self.criteria:
            if event_type not in self.criteria["event_type"]:
                return False
        
        if "entity_id" in self.criteria and hasattr(event_data, 'entity_id'):
            if event_data.entity_id not in self.criteria["entity_id"]:
                return False
        
        if "change_type" in self.criteria and hasattr(event_data, 'change_type'):
            if event_data.change_type not in self.criteria["change_type"]:
                return False
        
        return True


class FilteredEventBus(EventBus):
    """Event bus with filtering capabilities"""
    
    def __init__(self):
        super().__init__()
        self._filters: Dict[str, EventFilter] = {}
    
    def add_filter(self, subscriber_id: str, event_filter: EventFilter):
        """Add filter for specific subscriber"""
        self._filters[subscriber_id] = event_filter
    
    def remove_filter(self, subscriber_id: str):
        """Remove filter for subscriber"""
        if subscriber_id in self._filters:
            del self._filters[subscriber_id]
    
    def publish(self, event_type: str, event_data: Any):
        """Publish with filtering"""
        with self._lock:
            # Record event
            event_record = {
                "event_type": event_type,
                "timestamp": datetime.now(),
                "data": event_data,
                "subscribers_notified": 0,
                "delivery_failures": 0,
                "filtered_out": 0
            }
            
            self.metrics["events_published"] += 1
            
            # Notify subscribers with filtering
            if event_type in self._subscribers:
                active_subscribers = [s for s in self._subscribers[event_type] if s.active]
                
                for subscription in active_subscribers:
                    # Check filter
                    if subscription.subscriber_id in self._filters:
                        filter_obj = self._filters[subscription.subscriber_id]
                        if not filter_obj.matches(event_type, event_data):
                            event_record["filtered_out"] += 1
                            continue
                    
                    try:
                        subscription.callback(event_data)
                        event_record["subscribers_notified"] += 1
                        self.metrics["events_delivered"] += 1
                    except Exception as e:
                        event_record["delivery_failures"] += 1
                        self.metrics["failed_deliveries"] += 1
                        logger.error(f"Failed to deliver event {event_type} to "
                                   f"{subscription.subscriber_id}: {e}")
            
            # Store in history
            self._event_history.append(event_record)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]


class EventAggregator:
    """Aggregates related events for batch processing"""
    
    def __init__(self, aggregation_window: float = 1.0):
        self.aggregation_window = aggregation_window  # seconds
        self._pending_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._timers: Dict[str, threading.Timer] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.RLock()
    
    def register_aggregation(self, event_type: str, callback: Callable):
        """Register callback for aggregated events"""
        with self._lock:
            self._callbacks[event_type] = callback
    
    def add_event(self, event_type: str, event_data: Any):
        """Add event for aggregation"""
        with self._lock:
            event_record = {
                "timestamp": datetime.now(),
                "data": event_data
            }
            
            self._pending_events[event_type].append(event_record)
            
            # Reset timer
            if event_type in self._timers:
                self._timers[event_type].cancel()
            
            self._timers[event_type] = threading.Timer(
                self.aggregation_window,
                self._flush_events,
                args=[event_type]
            )
            self._timers[event_type].start()
    
    def _flush_events(self, event_type: str):
        """Flush aggregated events"""
        with self._lock:
            if event_type in self._pending_events and self._pending_events[event_type]:
                events = self._pending_events[event_type].copy()
                self._pending_events[event_type].clear()
                
                if event_type in self._callbacks:
                    try:
                        self._callbacks[event_type](events)
                    except Exception as e:
                        logger.error(f"Failed to process aggregated events for {event_type}: {e}")
    
    def flush_all(self):
        """Flush all pending events immediately"""
        with self._lock:
            for event_type in list(self._pending_events.keys()):
                if event_type in self._timers:
                    self._timers[event_type].cancel()
                self._flush_events(event_type)