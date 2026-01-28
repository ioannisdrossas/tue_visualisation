import pandas as pd
import numpy as np

class StaffingDataManager:
    """
    Service class to handle loading, cleaning, and processing 
    staffing and patient data.
    """
    
    def __init__(self, services_csv: str, schedule_csv: str, patients_csv: str):
        """
        Initializes the data manager and loads the benchmarks.

        Args:
            services_csv (str): Path to the services CSV file.
            schedule_csv (str): Path to the schedule CSV file.
            patients_csv (str): Path to the patients CSV file.
        """
        
        self.services_csv = services_csv
        self.schedule_csv = schedule_csv
        self.patients_csv = patients_csv
        
        # Define Staffing Ratio Benchmarks per department
        self.BENCHMARKS = {
            'ICU': {'nurse': 2, 'doctor': 14, 'alos_min': 2, 'alos_max': 4},
            'surgery': {'nurse': 6, 'doctor': 15, 'alos_min': 1, 'alos_max': 8},
            'general_medicine': {'nurse': 6, 'doctor': 15, 'alos_min': 4, 'alos_max': 6},
            'emergency': {'nurse': 4, 'doctor': None, 'alos_min': 4/24, 'alos_max': 6/24}
        }
        
        self.merged_df = None
        self.daily_merged = None
        self.available_services = []
        
        # Load data upon initialization
        self._load_and_process_data()

    def _load_and_process_data(self):
        """Internal method to load CSVs and perform feature engineering."""
        try:
            services_df = pd.read_csv(self.services_csv)
            staff_schedule_df = pd.read_csv(self.schedule_csv)
            patients_df = pd.read_csv(self.patients_csv)
        except FileNotFoundError:
            print("Error: CSV files not found. Please check the file paths.")
            return

        # Sanitize strings
        services_df['service'] = services_df['service'].str.strip()
        staff_schedule_df['service'] = staff_schedule_df['service'].str.strip()
        patients_df['service'] = patients_df['service'].str.strip()

        # prepare staff counts (weekly)
        present_staff = staff_schedule_df[staff_schedule_df['present'] == 1]
        staff_counts = present_staff.groupby(['week', 'service', 'role']).size().unstack(fill_value=0).reset_index()
        for role in ['doctor', 'nurse']:
            if role not in staff_counts.columns: staff_counts[role] = 0

        # prepare weekly data
        merged_df = pd.merge(services_df[['week', 'month', 'service', 'patients_admitted']], staff_counts, on=['week', 'service'], how='left')
        merged_df['doctor'] = merged_df['doctor'].fillna(0)
        merged_df['nurse'] = merged_df['nurse'].fillna(0)

        # Fix Month and Quarter Calculation
        merged_df['month'] = (pd.Timestamp('2025-01-01') + pd.to_timedelta((merged_df['week'] - 1) * 7, unit='D')).dt.month
        merged_df['quarter'] = ((merged_df['month'] - 1) // 3) + 1
        merged_df['week_of_month'] = merged_df.groupby('month')['week'].rank(method='dense').astype(int)

        # ALOS Calculation
        patients_df['arrival_date'] = pd.to_datetime(patients_df['arrival_date'])
        patients_df['departure_date'] = pd.to_datetime(patients_df['departure_date'])
        patients_df['los'] = (patients_df['departure_date'] - patients_df['arrival_date']).dt.days
        patients_df['week'] = patients_df['arrival_date'].dt.isocalendar().week

        alos_weekly = patients_df.groupby(['week', 'service'])['los'].mean().reset_index(name='avg_los')
        merged_df = pd.merge(merged_df, alos_weekly, on=['week', 'service'], how='left')
        merged_df['avg_los'] = merged_df['avg_los'].fillna(0)

        self.available_services = sorted(merged_df['service'].unique())

        def calculate_ratios(df, role):
            col_name = f'{role}_ratio'
            def get_ratio(row):
                patients = row['patients_admitted']
                staff = row[role]
                if staff > 0: 
                    return patients / staff
                else:
                    return 0 
            df[col_name] = df.apply(get_ratio, axis=1)
            return df

        merged_df = calculate_ratios(merged_df, 'nurse')
        merged_df = calculate_ratios(merged_df, 'doctor')
        
        self.merged_df = merged_df

        # Prepare daily data
        patients_df['day_name'] = patients_df['arrival_date'].dt.day_name()
        daily_patients = patients_df.groupby(['week', 'day_name', 'service']).size().reset_index(name='patients_admitted')
        daily_merged = pd.merge(daily_patients, staff_counts, on=['week', 'service'], how='left')
        daily_merged['doctor'] = daily_merged['doctor'].fillna(0)
        daily_merged['nurse'] = daily_merged['nurse'].fillna(0)

        daily_merged['month'] = (pd.Timestamp('2025-01-01') + pd.to_timedelta((daily_merged['week'] - 1) * 7, unit='D')).dt.month
        daily_merged['quarter'] = ((daily_merged['month'] - 1) // 3) + 1

        daily_merged = calculate_ratios(daily_merged, 'nurse')
        daily_merged = calculate_ratios(daily_merged, 'doctor')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_merged['day_name'] = pd.Categorical(daily_merged['day_name'], categories=days_order, ordered=True)
        daily_merged = daily_merged.sort_values(['week', 'day_name'])
        
        self.daily_merged = daily_merged

    def get_benchmarks(self, service:str) -> dict:
        """
        Retrieves the staffing benchmarks for a specific service.

        Args:
            service (str): The name of the service to retrieve benchmarks for.

        Returns:
            dict: A dictionary containing benchmark values (nurse ratio, doctor ratio, ALOS limits).
        """
        return self.BENCHMARKS.get(service, {})