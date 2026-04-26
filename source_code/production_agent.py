"""
Production Scheduling Agent

"""

import numpy as np
from typing import List, Dict, Tuple
from config import PRODUCTION_CAPACITY, SETUP_COST, PRODUCTS_TO_SCHEDULE, UNIT_PRICE, TARDINESS_PENALTY_RATE

class ProductionJob:
    """Represents a production order."""

    def __init__(self, job_id: int, product: str, quantity: int,
                 priority: int, due_date: int, unit_revenue: float):
        self.job_id = job_id
        self.product = product
        self.quantity = quantity
        self.priority = priority
        self.due_date = due_date
        self.unit_revenue = unit_revenue
        self.processing_time = max(1, quantity // PRODUCTION_CAPACITY)
        self.start_time = None
        self.completion_time = None
        self.tardiness = 0

    def __repr__(self):
        return (f"Job({self.job_id}, {self.product[:15]}, "
                f"qty={self.quantity}, due={self.due_date})")

class FIFOScheduler:
    """
    Baseline FIFO (First-In-First-Out) production scheduler.
    
    """

    def __init__(self):
        self.name = "FIFO Scheduler (Baseline)"
        self.schedule_log: List[Dict] = []

    def schedule(self, jobs: List[ProductionJob],
                 current_day: int) -> List[ProductionJob]:
        """Schedule jobs in arrival order."""
        scheduled = sorted(jobs, key=lambda j: j.job_id)
        time = current_day
        prev_product = None
        total_setup_cost = 0.0

        for job in scheduled:
            # Setup cost if product changes
            if prev_product and prev_product != job.product:
                total_setup_cost += SETUP_COST
                time += 1  # setup time

            job.start_time = time
            time += job.processing_time
            job.completion_time = time
            job.tardiness = max(0, job.completion_time - job.due_date)
            prev_product = job.product

        record = {
            'day': current_day,
            'scheduler': self.name,
            'n_jobs': len(jobs),
            'makespan': time - current_day,
            'total_tardiness': sum(j.tardiness for j in scheduled),
            'total_setup_cost': total_setup_cost,
            'jobs': [str(j) for j in scheduled]
        }
        self.schedule_log.append(record)
        return scheduled

    def compute_cost(self, jobs: List[ProductionJob]) -> float:
        scheduled = sorted(jobs, key=lambda j: j.job_id)
        cost = 0.0
        prev_product = None
        for job in scheduled:
            if prev_product and prev_product != job.product:
                cost += SETUP_COST
            cost += job.tardiness * job.unit_revenue * TARDINESS_PENALTY_RATE  # tardiness penalty
            prev_product = job.product
        return cost

class OptimizedScheduler:
    """
    Agentic production scheduler using Shortest Processing Time (SPT)
    with weighted tardiness minimization and setup cost reduction.
    
    """

    def __init__(self):
        self.name = "SPT + Weighted Tardiness Optimizer (Agentic)"
        self.schedule_log: List[Dict] = []

    def schedule(self, jobs: List[ProductionJob],
                 current_day: int) -> List[ProductionJob]:
        """
        Optimized schedule:
        1. Group jobs by product to minimize setups
        2. Within groups: sort by (tardiness_risk / processing_time) ratio
        """
        if not jobs:
            return []

        # Score each job: urgency = (due_date - current_day) / processing_time
        # Lower score = more urgent
        def urgency(j):
            slack = max(0, j.due_date - current_day)
            return slack / j.processing_time

        # Group by product to reduce setups, then sort by urgency within group
        from itertools import groupby
        # First, find optimal product sequence (greedy nearest-neighbor)
        product_groups: Dict[str, List[ProductionJob]] = {}
        for job in jobs:
            product_groups.setdefault(job.product, []).append(job)

        # Sort within each product group by urgency
        for p in product_groups:
            product_groups[p].sort(key=urgency)

        # Greedy sequence: start with most urgent product group
        product_urgency = {
            p: min(urgency(j) for j in jlist)
            for p, jlist in product_groups.items()
        }
        product_order = sorted(product_urgency.keys(), key=lambda p: product_urgency[p])

        scheduled = []
        for p in product_order:
            scheduled.extend(product_groups[p])

        time = current_day
        prev_product = None
        total_setup_cost = 0.0

        for job in scheduled:
            if prev_product and prev_product != job.product:
                total_setup_cost += SETUP_COST
                time += 1
            job.start_time = time
            time += job.processing_time
            job.completion_time = time
            job.tardiness = max(0, job.completion_time - job.due_date)
            prev_product = job.product

        record = {
            'day': current_day,
            'scheduler': self.name,
            'n_jobs': len(jobs),
            'makespan': time - current_day,
            'total_tardiness': sum(j.tardiness for j in scheduled),
            'total_setup_cost': total_setup_cost,
            'jobs': [str(j) for j in scheduled]
        }
        self.schedule_log.append(record)
        return scheduled

    def compute_cost(self, jobs: List[ProductionJob]) -> float:
        scheduled = self.schedule(jobs, 0)
        cost = 0.0
        prev_product = None
        for job in scheduled:
            if prev_product and prev_product != job.product:
                cost += SETUP_COST
            cost += job.tardiness * job.unit_revenue * TARDINESS_PENALTY_RATE
            prev_product = job.product
        return cost

def generate_production_jobs(day: int, demand_forecast: float,
                              rng: np.random.Generator,
                              n_products: int = PRODUCTS_TO_SCHEDULE
                              ) -> List[ProductionJob]:
    """Generate a batch of production jobs for scheduling."""
    products = [f"Product_{chr(65 + i)}" for i in range(n_products)]
    revenues = [UNIT_PRICE * rng.uniform(0.8, 1.2) for _ in range(n_products)]
    jobs = []
    job_id = day * 100

    for i, (prod, rev) in enumerate(zip(products, revenues)):
        qty = max(10, int(rng.poisson(demand_forecast / n_products)))
        due = day + int(rng.uniform(2, 8))
        priority = int(rng.integers(1, 4))
        jobs.append(ProductionJob(job_id + i, prod, qty, priority, due, rev))

    return jobs
