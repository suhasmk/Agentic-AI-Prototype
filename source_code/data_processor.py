"""
Data Processing Module
Loads and preprocesses the DataCo Supply Chain Dataset.
Extracts historical demand, lead times, and logistics features.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class DataCoProcessor:
    """Preprocesses DataCo Supply Chain dataset for simulation calibration."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.daily_demand = None
        self.lead_time_stats = {}
        self.product_stats = {}
        self.logistics_stats = {}
        self.top_products = []

    def load(self):
        """Load and clean dataset."""
        self.df = pd.read_csv(self.filepath, encoding='latin1')

        # Parse dates
        self.df['order_date'] = pd.to_datetime(
            self.df['order date (DateOrders)'], format='%m/%d/%Y %H:%M', errors='coerce')
        self.df['ship_date'] = pd.to_datetime(
            self.df['shipping date (DateOrders)'], format='%m/%d/%Y %H:%M', errors='coerce')

        # Drop rows with missing dates
        self.df.dropna(subset=['order_date'], inplace=True)
        self.df['year_month'] = self.df['order_date'].dt.to_period('M')
        self.df['day_of_year'] = self.df['order_date'].dt.dayofyear

        print(f"[DataProcessor] Loaded {len(self.df):,} records spanning "
              f"{self.df['order_date'].min().date()} to {self.df['order_date'].max().date()}")
        return self

    def extract_demand(self):
        """Aggregate daily demand and compute Poisson lambda."""
        daily = (self.df.groupby(self.df['order_date'].dt.date)['Order Item Quantity']
                 .sum().reset_index())
        daily.columns = ['date', 'demand']
        self.daily_demand = daily

        self.demand_lambda = float(daily['demand'].mean())
        self.demand_std = float(daily['demand'].std())
        print(f"[DataProcessor] Daily demand  mean={self.demand_lambda:.1f}, "
              f"std={self.demand_std:.1f} (Poisson lambda calibrated)")
        return self

    def extract_lead_times(self):
        """Extract lead time distribution from actual vs scheduled shipping."""
        lt = self.df[['Days for shipping (real)',
                      'Days for shipment (scheduled)']].dropna()
        self.lead_time_stats = {
            'mean': float(lt['Days for shipping (real)'].mean()),
            'std': float(lt['Days for shipping (real)'].std()),
            'scheduled_mean': float(lt['Days for shipment (scheduled)'].mean()),
        }
        print(f"[DataProcessor] Lead time  mean={self.lead_time_stats['mean']:.2f}d, "
              f"std={self.lead_time_stats['std']:.2f}d")
        return self

    def extract_product_stats(self):
        """Get top products by revenue for simulation."""
        prod = (self.df.groupby('Product Name')
                .agg(total_qty=('Order Item Quantity', 'sum'),
                     avg_price=('Product Price', 'mean'),
                     avg_profit_ratio=('Order Item Profit Ratio', 'mean'),
                     total_revenue=('Sales', 'sum'))
                .reset_index()
                .sort_values('total_revenue', ascending=False))
        self.top_products = prod.head(10)['Product Name'].tolist()
        self.product_stats = prod.set_index('Product Name').to_dict('index')
        print(f"[DataProcessor] Top product: {self.top_products[0]}")
        return self

    def extract_logistics_stats(self):
        """Compute shipping mode usage and late delivery rates."""
        lg = (self.df.groupby('Shipping Mode')
              .agg(count=('Order Id', 'count'),
                   late_rate=('Late_delivery_risk', 'mean'),
                   avg_days=('Days for shipping (real)', 'mean'))
              .reset_index())
        self.logistics_stats = lg.set_index('Shipping Mode').to_dict('index')
        late_overall = float(self.df['Late_delivery_risk'].mean())
        cancelled = (self.df['Order Status'] == 'SUSPECTED_FRAUD').sum()
        disruption_proxy = float((self.df['Delivery Status'] == 'Shipping canceled').mean())
        self.disruption_rate = disruption_proxy
        print(f"[DataProcessor] Overall late delivery rate: {late_overall:.1%}, "
              f"disruption proxy: {disruption_proxy:.1%}")
        return self

    def get_monthly_demand_series(self):
        """Return monthly aggregated demand series for forecasting."""
        monthly = (self.df.groupby('year_month')['Order Item Quantity']
                   .sum().reset_index())
        monthly.columns = ['period', 'demand']
        return monthly

    def run_all(self):
        """Run full preprocessing pipeline."""
        return (self.load()
                .extract_demand()
                .extract_lead_times()
                .extract_product_stats()
                .extract_logistics_stats())
