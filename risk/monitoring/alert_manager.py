import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import threading
import structlog

logger = structlog.get_logger()


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    timestamp: float
    level: AlertLevel
    metric_name: str
    value: float
    message: str


class AlertManager:
    """
    Handles real-time risk alert generation, logging, and external notification routing.
    Includes rate-limiting/throttling mechanism to prevent spam.
    """

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.history: List[Alert] = []
        self._last_alert_times: Dict[str, float] = {}  # key: f"{metric_name}_{level}" -> timestamp
        self._lock = threading.Lock()
        logger.info("AlertManager initialized", cooldown_seconds=cooldown_seconds)

    def trigger(
        self,
        level: AlertLevel,
        metric_name: str,
        value: float,
        message: str
    ) -> bool:
        """
        Triggers a risk alert. If the exact same alert was triggered within the
        cooldown period, it is throttled.
        
        Returns:
            True if the alert was successfully processed and logged, False if throttled.
        """
        now = time.time()
        cooldown_key = f"{metric_name}_{level.value}"

        with self._lock:
            last_time = self._last_alert_times.get(cooldown_key, 0.0)
            if now - last_time < self.cooldown_seconds:
                # Throttled
                logger.debug(
                    "Alert throttled",
                    metric_name=metric_name,
                    level=level.value,
                    seconds_left=self.cooldown_seconds - (now - last_time)
                )
                return False

            # Process alert
            alert = Alert(
                timestamp=now,
                level=level,
                metric_name=metric_name,
                value=value,
                message=message
            )
            self.history.append(alert)
            self._last_alert_times[cooldown_key] = now

        # Log according to severity
        log_kwargs = {
            "metric": metric_name,
            "value": value,
            "message": message,
            "timestamp": alert.timestamp
        }
        if level == AlertLevel.CRITICAL:
            logger.error("CRITICAL RISK ALERT TRIGGERED", **log_kwargs)
            # In a live setup, routing to PagerDuty or critical Slack channels happens here.
        elif level == AlertLevel.WARNING:
            logger.warning("WARNING RISK ALERT TRIGGERED", **log_kwargs)
        else:
            logger.info("INFO Alert logged", **log_kwargs)

        return True

    def get_alerts(self, min_level: Optional[AlertLevel] = None) -> List[Alert]:
        """
        Retrieves the alert history, optionally filtered by minimum alert level.
        """
        with self._lock:
            if min_level is None:
                return list(self.history)
            
            level_ranks = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1, AlertLevel.CRITICAL: 2}
            target_rank = level_ranks.get(min_level, 0)
            
            return [
                alert for alert in self.history
                if level_ranks.get(alert.level, 0) >= target_rank
            ]

    def clear_cooldowns(self) -> None:
        """
        Resets all throttling cooldowns.
        """
        with self._lock:
            self._last_alert_times.clear()
            logger.debug("Alert cooldown timers cleared")
