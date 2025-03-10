import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitor and track system performance metrics
    """
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the performance monitor
        
        Args:
            log_dir (str): Directory to store performance logs
        """
        self.log_dir = log_dir
        self.metrics = {
            "query_count": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "response_times": [],
            "agent_usage": {
                "directory_agent": 0,
                "finder_agent": 0,
                "cashflow_agent": 0,
                "screener_agent": 0
            },
            "lamini_api_calls": 0,  # Added for Lamini API tracking
            "lamini_errors": 0,     # Added for Lamini error tracking
            "avg_lamini_response_time": 0,  # Added for Lamini response time tracking
            "errors": []
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics file
        self.metrics_file = os.path.join(log_dir, f"metrics_{datetime.now().strftime('%Y%m%d')}.json")
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file if it exists"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded metrics from {self.metrics_file}")
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def start_query(self) -> float:
        """
        Start timing a query
        
        Returns:
            float: Start time
        """
        return time.time()
    
    def end_query(self, start_time: float, agent: str, success: bool, error_message: Optional[str] = None):
        """
        End timing a query and record metrics
        
        Args:
            start_time (float): Start time from start_query()
            agent (str): Agent that processed the query
            success (bool): Whether the query was successful
            error_message (Optional[str]): Error message if the query failed
        """
        end_time = time.time()
        response_time = end_time - start_time
        
        # Update metrics
        self.metrics["query_count"] += 1
        self.metrics["response_times"].append(response_time)
        
        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1
            if error_message:
                self.metrics["errors"].append({
                    "timestamp": datetime.now().isoformat(),
                    "agent": agent,
                    "message": error_message
                })
        
        # Update agent usage
        if agent in self.metrics["agent_usage"]:
            self.metrics["agent_usage"][agent] += 1
        
        # Save metrics
        self._save_metrics()
        
        # Log response time
        logger.info(f"Query processed by {agent} in {response_time:.2f}s (success: {success})")
    
    def track_lamini_call(self, response_time: float, success: bool):
        """
        Track a Lamini API call
        
        Args:
            response_time (float): Response time in seconds
            success (bool): Whether the call was successful
        """
        self.metrics["lamini_api_calls"] += 1
        
        if not success:
            self.metrics["lamini_errors"] += 1
        
        # Update average response time
        current_avg = self.metrics["avg_lamini_response_time"]
        current_count = self.metrics["lamini_api_calls"]
        
        # Calculate new average
        self.metrics["avg_lamini_response_time"] = (current_avg * (current_count - 1) + response_time) / current_count
        
        # Save metrics
        self._save_metrics()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        # Calculate average response time
        avg_response_time = np.mean(self.metrics["response_times"]) if self.metrics["response_times"] else 0
        
        # Calculate error rate
        error_rate = (self.metrics["failed_queries"] / self.metrics["query_count"]) * 100 if self.metrics["query_count"] > 0 else 0
        
        # Calculate Lamini error rate
        lamini_error_rate = (self.metrics["lamini_errors"] / self.metrics["lamini_api_calls"]) * 100 if self.metrics["lamini_api_calls"] > 0 else 0
        
        return {
            "query_count": self.metrics["query_count"],
            "successful_queries": self.metrics["successful_queries"],
            "failed_queries": self.metrics["failed_queries"],
            "avg_response_time": avg_response_time,
            "error_rate": error_rate,
            "agent_usage": self.metrics["agent_usage"],
            "lamini_api_calls": self.metrics["lamini_api_calls"],
            "lamini_errors": self.metrics["lamini_errors"],
            "lamini_error_rate": lamini_error_rate,
            "avg_lamini_response_time": self.metrics["avg_lamini_response_time"]
        }
    
    def generate_report(self) -> str:
        """
        Generate a performance report
        
        Returns:
            str: Performance report
        """
        metrics = self.get_performance_metrics()
        
        report = [
            "# Performance Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"Total Queries: {metrics['query_count']}",
            f"Success Rate: {metrics['success_rate']:.2f}%" if metrics['success_rate'] is not None else "Success Rate: N/A",
            f"Average Response Time: {metrics['avg_response_time']:.2f}s" if metrics['avg_response_time'] is not None else "Average Response Time: N/A",
            "",
            "## Agent Usage",
        ]
        
        for agent, count in metrics["agent_usage"].items():
            distribution = metrics["agent_distribution"][agent]
            report.append(f"- {agent}: {count} queries ({distribution:.2f}%)")
        
        report.extend([
            "",
            "## Response Time Percentiles",
            f"P95: {metrics['p95_response_time']:.2f}s" if metrics['p95_response_time'] is not None else "P95: N/A",
            f"P99: {metrics['p99_response_time']:.2f}s" if metrics['p99_response_time'] is not None else "P99: N/A",
            "",
            "## Recent Errors",
        ])
        
        if metrics["errors"]:
            for error in metrics["errors"][-5:]:
                report.append(f"- {error['timestamp']}: {error['agent']} - {error['message']}")
        else:
            report.append("No recent errors")
        
        return "\n".join(report)

# Create a singleton instance
performance_monitor = PerformanceMonitor() 