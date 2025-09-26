"""
Time management and timezone utilities.
"""
from datetime import datetime
import pytz
from typing import Dict, Optional

class TimeManager:
    def __init__(self):
        self.default_timezone = pytz.UTC
        
    def convert_time(
        self,
        time: datetime,
        from_tz: str,
        to_tz: str
    ) -> datetime:
        """
        Convert time between timezones.
        
        Args:
            time: Datetime to convert
            from_tz: Source timezone
            to_tz: Target timezone
            
        Returns:
            Converted datetime
        """
        source = pytz.timezone(from_tz)
        target = pytz.timezone(to_tz)
        
        # Ensure time is timezone-aware
        if time.tzinfo is None:
            time = source.localize(time)
            
        return time.astimezone(target)
        
    def get_current_time(
        self,
        timezone: Optional[str] = None
    ) -> datetime:
        """
        Get current time in specified timezone.
        
        Args:
            timezone: Optional timezone name
            
        Returns:
            Current datetime in requested timezone
        """
        now = datetime.now(self.default_timezone)
        if timezone:
            return now.astimezone(pytz.timezone(timezone))
        return now