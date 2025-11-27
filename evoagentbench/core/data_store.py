"""
Data Storage Layer for EvoAgentBench

Implements the central data repository using DuckDB (or SQLite for simpler deployments).
Stores all persistent data: Genomes, Tasks, Runs, Metrics, and Traces.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib


class DataStore:
    """
    Central data storage layer for EvoAgentBench.
    Uses SQLite as the embedded database (can be upgraded to DuckDB later).
    """
    
    def __init__(self, db_path: str = "evoagentbench.db"):
        """
        Initialize the data store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Create all required tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Genomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genomes (
                genome_id TEXT PRIMARY KEY,
                parent_id TEXT,
                generation INTEGER NOT NULL,
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES genomes(genome_id)
            )
        """)
        
        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                category TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                checker_type TEXT NOT NULL,
                task_spec_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (task_id, version)
            )
        """)
        
        # Runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                trace_id TEXT UNIQUE NOT NULL,
                genome_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_version INTEGER NOT NULL,
                generation_id INTEGER,
                run_seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                start_timestamp TIMESTAMP NOT NULL,
                end_timestamp TIMESTAMP,
                run_manifest_json TEXT NOT NULL,
                FOREIGN KEY (genome_id) REFERENCES genomes(genome_id),
                FOREIGN KEY (task_id, task_version) REFERENCES tasks(task_id, version)
            )
        """)
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                is_hard_metric INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, metric_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Traces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                input_hash TEXT,
                output_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trace_id, step_index),
                FOREIGN KEY (trace_id) REFERENCES runs(trace_id)
            )
        """)
        
        # Ablations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ablations (
                ablation_id TEXT PRIMARY KEY,
                base_genome_id TEXT NOT NULL,
                factor_key TEXT NOT NULL,
                factor_value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (base_genome_id) REFERENCES genomes(genome_id)
            )
        """)
        
        # Generations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                best_genome_id TEXT,
                avg_fitness REAL,
                population_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (best_genome_id) REFERENCES genomes(genome_id)
            )
        """)
        
        # Create indexes for performance
        self._create_indexes(cursor)
        
        self.conn.commit()
    
    def _create_indexes(self, cursor):
        """Create indexes for common queries."""
        indexes = [
            ("idx_genomes_generation", "genomes", "generation"),
            ("idx_genomes_config_hash", "genomes", "config_hash"),
            ("idx_tasks_category", "tasks", "category"),
            ("idx_tasks_difficulty", "tasks", "difficulty"),
            ("idx_runs_genome_id", "runs", "genome_id"),
            ("idx_runs_task_id", "runs", "task_id"),
            ("idx_runs_status", "runs", "status"),
            ("idx_metrics_metric_name", "metrics", "metric_name"),
            ("idx_metrics_metric_value", "metrics", "metric_value"),
            ("idx_traces_trace_id", "traces", "trace_id"),
            ("idx_traces_event_type", "traces", "event_type"),
            ("idx_ablations_base_genome", "ablations", "base_genome_id"),
            ("idx_ablations_factor_key", "ablations", "factor_key"),
        ]
        
        for index_name, table, column in indexes:
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})
                """)
            except sqlite3.OperationalError:
                # Index might already exist
                pass
    
    def _compute_hash(self, data: Any) -> str:
        """Compute SHA256 hash of JSON-serializable data."""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    # ========== Genome Operations ==========
    
    def save_genome(self, genome_config: Dict[str, Any]) -> str:
        """
        Save a genome configuration.
        
        Args:
            genome_config: Genome configuration dictionary
            
        Returns:
            genome_id: The ID of the saved genome
        """
        genome_id = genome_config.get("genome_id") or str(uuid.uuid4())
        config_hash = self._compute_hash(genome_config)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO genomes 
            (genome_id, parent_id, generation, config_hash, config_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            genome_id,
            genome_config.get("parent_id"),
            genome_config.get("generation", 0),
            config_hash,
            json.dumps(genome_config)
        ))
        self.conn.commit()
        return genome_id
    
    def get_genome(self, genome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a genome by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT config_json FROM genomes WHERE genome_id = ?", (genome_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def get_genomes_by_generation(self, generation: int) -> List[Dict[str, Any]]:
        """Get all genomes in a specific generation."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT config_json FROM genomes WHERE generation = ?", (generation,))
        return [json.loads(row[0]) for row in cursor.fetchall()]
    
    # ========== Task Operations ==========
    
    def save_task(self, task_spec: Dict[str, Any]):
        """Save a task specification."""
        cursor = self.conn.cursor()
        
        # Extract category as comma-separated string for indexing
        categories = task_spec.get("category", [])
        category_str = ",".join(categories) if isinstance(categories, list) else str(categories)
        
        cursor.execute("""
            INSERT OR REPLACE INTO tasks 
            (task_id, version, category, difficulty, checker_type, task_spec_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            task_spec["task_id"],
            task_spec["version"],
            category_str,
            task_spec["difficulty"],
            task_spec["checker_type"],
            json.dumps(task_spec)
        ))
        self.conn.commit()
    
    def get_task(self, task_id: str, version: int) -> Optional[Dict[str, Any]]:
        """Retrieve a task by ID and version."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT task_spec_json FROM tasks WHERE task_id = ? AND version = ?",
            (task_id, version)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT task_spec_json FROM tasks")
        return [json.loads(row[0]) for row in cursor.fetchall()]
    
    # ========== Run Operations ==========
    
    def save_run(self, run_manifest: Dict[str, Any], status: str = "PENDING", 
                 end_timestamp: Optional[datetime] = None) -> str:
        """
        Save a run record.
        
        Args:
            run_manifest: Complete run manifest dictionary
            status: Run status (PENDING, SUCCESS, FAILURE, BUDGET_EXCEEDED, TIMEOUT, etc.)
            end_timestamp: Optional end timestamp
            
        Returns:
            run_id: The ID of the saved run
        """
        run_id = run_manifest.get("run_id") or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO runs 
            (run_id, trace_id, genome_id, task_id, task_version, generation_id, 
             run_seed, status, start_timestamp, end_timestamp, run_manifest_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            trace_id,
            run_manifest["genome_id"],
            run_manifest["task_id"],
            run_manifest["task_version"],
            run_manifest.get("generation_id"),
            run_manifest["run_seed"],
            status,
            run_manifest.get("start_timestamp", datetime.utcnow().isoformat()),
            end_timestamp.isoformat() if end_timestamp else None,
            json.dumps(run_manifest)
        ))
        self.conn.commit()
        return run_id
    
    def update_run_status(self, run_id: str, status: str, 
                         end_timestamp: Optional[datetime] = None):
        """Update the status of a run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE runs 
            SET status = ?, end_timestamp = ?
            WHERE run_id = ?
        """, (
            status,
            end_timestamp.isoformat() if end_timestamp else None,
            run_id
        ))
        self.conn.commit()
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a run by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_trace_id(self, run_id: str) -> Optional[str]:
        """Get the trace_id for a run."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT trace_id FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    # ========== Metric Operations ==========
    
    def save_metric(self, run_id: str, metric_name: str, metric_value: float, 
                   is_hard_metric: bool = True):
        """Save a metric for a run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO metrics 
            (run_id, metric_name, metric_value, is_hard_metric)
            VALUES (?, ?, ?, ?)
        """, (run_id, metric_name, metric_value, 1 if is_hard_metric else 0))
        self.conn.commit()
    
    def save_metrics(self, run_id: str, metrics: Dict[str, float], 
                    is_hard_metric: bool = True):
        """Save multiple metrics for a run."""
        for metric_name, metric_value in metrics.items():
            self.save_metric(run_id, metric_name, metric_value, is_hard_metric)
    
    def get_metrics(self, run_id: str) -> Dict[str, float]:
        """Get all metrics for a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT metric_name, metric_value FROM metrics WHERE run_id = ?",
            (run_id,)
        )
        return {row[0]: row[1] for row in cursor.fetchall()}
    
    def get_fitness_by_generation(self, generation: int, 
                                  metric_name: str = "weighted_fitness") -> List[Tuple[str, float]]:
        """
        Get fitness scores for all genomes in a generation.
        Used for evolutionary selection.
        
        Returns:
            List of (genome_id, fitness) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT g.genome_id, AVG(m.metric_value) AS fitness
            FROM genomes g
            JOIN runs r ON g.genome_id = r.genome_id
            JOIN metrics m ON r.run_id = m.run_id
            WHERE g.generation = ? AND m.metric_name = ?
            GROUP BY g.genome_id
        """, (generation, metric_name))
        return [(row[0], row[1]) for row in cursor.fetchall()]
    
    # ========== Trace Operations ==========
    
    def save_trace_event(self, trace_id: str, step_index: int, timestamp: float,
                        event_type: str, payload: Dict[str, Any],
                        input_hash: Optional[str] = None,
                        output_hash: Optional[str] = None):
        """Save a single trace event."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO traces 
            (trace_id, step_index, timestamp, event_type, payload, input_hash, output_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            trace_id,
            step_index,
            timestamp,
            event_type,
            json.dumps(payload),
            input_hash,
            output_hash
        ))
        self.conn.commit()
    
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Retrieve the full trace for a run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT payload, event_type, step_index, timestamp, input_hash, output_hash
            FROM traces
            WHERE trace_id = ?
            ORDER BY step_index ASC
        """, (trace_id,))
        
        return [
            {
                "step_index": row[2],
                "timestamp": row[3],
                "event_type": row[1],
                "payload": json.loads(row[0]),
                "input_hash": row[4],
                "output_hash": row[5]
            }
            for row in cursor.fetchall()
        ]
    
    # ========== Generation Operations ==========
    
    def save_generation(self, generation_id: int, start_time: datetime,
                       best_genome_id: Optional[str] = None,
                       avg_fitness: Optional[float] = None,
                       population_size: Optional[int] = None):
        """Save generation summary."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO generations 
            (generation_id, start_time, best_genome_id, avg_fitness, population_size)
            VALUES (?, ?, ?, ?, ?)
        """, (generation_id, start_time.isoformat(), best_genome_id, avg_fitness, population_size))
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
