"""
ToonDB Agent Memory System - Performance Tracker
Tracks latency metrics across agent operations
"""
import time
from typing import List
from dataclasses import dataclass
import statistics


@dataclass
class CycleMetrics:
    """Metrics for a single agent cycle"""
    write_latency_ms: float
    read_latency_ms: float
    assemble_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    num_cycles: int
    
    # Per-operation latencies (P50, P95, P99, P99.9)
    write_p50: float
    write_p95: float
    write_p99: float
    write_p999: float
    
    read_p50: float
    read_p95: float
    read_p99: float
    read_p999: float
    
    assemble_p50: float
    assemble_p95: float
    assemble_p99: float
    assemble_p999: float
    
    llm_p50: float
    llm_p95: float
    llm_p99: float
    llm_p999: float
    
    # End-to-end latencies
    total_p50: float
    total_p95: float
    total_p99: float
    total_p999: float
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║           ToonDB Agent Performance Report                    ║
╚══════════════════════════════════════════════════════════════╝

📊 Cycles Analyzed: {self.num_cycles}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 END-TO-END LATENCY (The Number That Matters)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  P50  (median): {self.total_p50:8.2f} ms
  P95  :          {self.total_p95:8.2f} ms
  P99  :          {self.total_p99:8.2f} ms ⭐ KEY METRIC
  P99.9:          {self.total_p999:8.2f} ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 OPERATION BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write (Store Observation):
  P50:  {self.write_p50:8.2f} ms  |  P99: {self.write_p99:8.2f} ms

Read (Vector Search):
  P50:  {self.read_p50:8.2f} ms  |  P99: {self.read_p99:8.2f} ms

Assemble (Build Context):
  P50:  {self.assemble_p50:8.2f} ms  |  P99: {self.assemble_p99:8.2f} ms

LLM (Generate Response):
  P50:  {self.llm_p50:8.2f} ms  |  P99: {self.llm_p99:8.2f} ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class PerformanceTracker:
    """
    Tracks latency metrics for agent operations
    
    Collects:
    - Write latency (storing observations)
    - Read latency (vector search + retrieval)
    - Assemble latency (context building)
    - LLM latency (response generation)
    - Total end-to-end latency
    """
    
    def __init__(self):
        self.cycles: List[CycleMetrics] = []
    
    def record_cycle(
        self,
        write_ms: float,
        read_ms: float,
        assemble_ms: float,
        llm_ms: float
    ):
        """Record a complete cycle's latencies"""
        total_ms = write_ms + read_ms + assemble_ms + llm_ms
        
        self.cycles.append(CycleMetrics(
            write_latency_ms=write_ms,
            read_latency_ms=read_ms,
            assemble_latency_ms=assemble_ms,
            llm_latency_ms=llm_ms,
            total_latency_ms=total_ms
        ))
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def get_report(self) -> PerformanceReport:
        """Generate comprehensive latency report with percentiles"""
        if not self.cycles:
            # Return zeroed report
            return PerformanceReport(
                num_cycles=0,
                write_p50=0, write_p95=0, write_p99=0, write_p999=0,
                read_p50=0, read_p95=0, read_p99=0, read_p999=0,
                assemble_p50=0, assemble_p95=0, assemble_p99=0, assemble_p999=0,
                llm_p50=0, llm_p95=0, llm_p99=0, llm_p999=0,
                total_p50=0, total_p95=0, total_p99=0, total_p999=0
            )
        
        # Extract latencies by operation
        write_latencies = [c.write_latency_ms for c in self.cycles]
        read_latencies = [c.read_latency_ms for c in self.cycles]
        assemble_latencies = [c.assemble_latency_ms for c in self.cycles]
        llm_latencies = [c.llm_latency_ms for c in self.cycles]
        total_latencies = [c.total_latency_ms for c in self.cycles]
        
        return PerformanceReport(
            num_cycles=len(self.cycles),
            
            # Write percentiles
            write_p50=self._percentile(write_latencies, 50),
            write_p95=self._percentile(write_latencies, 95),
            write_p99=self._percentile(write_latencies, 99),
            write_p999=self._percentile(write_latencies, 99.9),
            
            # Read percentiles
            read_p50=self._percentile(read_latencies, 50),
            read_p95=self._percentile(read_latencies, 95),
            read_p99=self._percentile(read_latencies, 99),
            read_p999=self._percentile(read_latencies, 99.9),
            
            # Assemble percentiles
            assemble_p50=self._percentile(assemble_latencies, 50),
            assemble_p95=self._percentile(assemble_latencies, 95),
            assemble_p99=self._percentile(assemble_latencies, 99),
            assemble_p999=self._percentile(assemble_latencies, 99.9),
            
            # LLM percentiles
            llm_p50=self._percentile(llm_latencies, 50),
            llm_p95=self._percentile(llm_latencies, 95),
            llm_p99=self._percentile(llm_latencies, 99),
            llm_p999=self._percentile(llm_latencies, 99.9),
            
            # Total percentiles
            total_p50=self._percentile(total_latencies, 50),
            total_p95=self._percentile(total_latencies, 95),
            total_p99=self._percentile(total_latencies, 99),
            total_p999=self._percentile(total_latencies, 99.9)
        )
