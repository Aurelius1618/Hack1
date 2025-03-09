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
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate average response time
        if metrics["response_times"]:
            metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
            metrics["p95_response_time"] = np.percentile(metrics["response_times"], 95) if len(metrics["response_times"]) > 10 else None
            metrics["p99_response_time"] = np.percentile(metrics["response_times"], 99) if len(metrics["response_times"]) > 100 else None
        else:
            metrics["avg_response_time"] = None
            metrics["p95_response_time"] = None
            metrics["p99_response_time"] = None
        
        # Calculate success rate
        if metrics["query_count"] > 0:
            metrics["success_rate"] = (metrics["successful_queries"] / metrics["query_count"]) * 100
        else:
            metrics["success_rate"] = None
        
        # Calculate agent distribution
        if metrics["query_count"] > 0:
            metrics["agent_distribution"] = {
                agent: (count / metrics["query_count"]) * 100
                for agent, count in metrics["agent_usage"].items()
            }
        else:
            metrics["agent_distribution"] = {agent: 0 for agent in metrics["agent_usage"]}
        
        # Limit number of response times and errors to return
        metrics["response_times"] = metrics["response_times"][-100:]
        metrics["errors"] = metrics["errors"][-20:]
        
        return metrics
    
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