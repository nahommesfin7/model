import pandas as pd
from datetime import datetime

class DynamicPricing:
    def __init__(self, data=None):
        self.data = data

    def calculate_bike_share_cost(self, duration_minutes, free_minutes=15, base_rate=0.45, surge_multiplier=1.0):
        """
        Calculate the cost of a bike-sharing trip using the "First X minutes free" pricing method
        with dynamic surge pricing.
    
        :param duration_minutes: The total duration of the trip in minutes.
        :param free_minutes: The number of free minutes. Default is 15 minutes.
        :param base_rate: The base rate charged per minute after the free period. Default is $0.45 per minute.
        :param surge_multiplier: Multiplier applied during surge pricing. Default is 1.0 (no surge).
        :return: The total cost of the trip.
        """
        effective_rate = base_rate * surge_multiplier
        if duration_minutes <= free_minutes:
            return 0.0
        else:
            chargeable_minutes = duration_minutes - free_minutes
            return chargeable_minutes * effective_rate

    def convert_to_minutes(self, duration):
        """
        Convert a given duration to minutes.
        
        :param duration: The duration in any format.
        :return: The duration in minutes.
        """
        if isinstance(duration, (int, float)):
            # If duration is already a number, assume it's in minutes
            return duration
        elif isinstance(duration, str):
            # Attempt to parse the duration if it's a string
            try:
                duration_td = pd.to_timedelta(duration)
                return duration_td.total_seconds() / 60.0
            except ValueError:
                raise ValueError(f"Cannot convert duration '{duration}' to minutes.")
        else:
            raise ValueError(f"Unsupported duration format: {duration}")

    def calculate_row_cost(self, row):
        # Convert the duration to minutes if necessary
        duration_minutes = self.convert_to_minutes(row['Duration'])
        surge_multiplier = 1.5 if row['Is Peak Time'] else 1.0
        return self.calculate_bike_share_cost(duration_minutes, surge_multiplier=surge_multiplier)

    def is_peak_time(self, timestamp):
        morning_peak_start = timestamp.replace(hour=7, minute=0, second=0, microsecond=0)
        morning_peak_end = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
        evening_peak_start = timestamp.replace(hour=17, minute=0, second=0, microsecond=0)
        evening_peak_end = timestamp.replace(hour=19, minute=0, second=0, microsecond=0)
        return morning_peak_start <= timestamp <= morning_peak_end or evening_peak_start <= timestamp <= evening_peak_end

    def processing(self):
        if self.data is not None:
            # Convert 'Start Time' to datetime
            self.data['Start Time'] = pd.to_datetime(self.data['Start Time'])
            
            # Convert 'Duration' to numeric, forcing errors to NaN and then filling NaNs with 0
            self.data['Duration'] = pd.to_numeric(self.data['Duration'], errors='coerce').fillna(0).astype(int)
            
            # Determine if the time is peak time
            self.data['Is Peak Time'] = self.data['Start Time'].apply(self.is_peak_time)
            
            # Calculate cost
            self.data['Cost'] = self.data.apply(self.calculate_row_cost, axis=1)
            return self.data
        else:
            raise ValueError("No data provided")
