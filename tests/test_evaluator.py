"""
Tests for Evaluator.
"""

import unittest

from evoagentbench.evaluation.evaluator import Evaluator
from evoagentbench.core.data_store import DataStore
import tempfile
import os


class TestEvaluator(unittest.TestCase):
    """Test cases for Evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.data_store = DataStore(self.temp_db.name)
        self.evaluator = Evaluator(self.data_store)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.data_store.close()
        os.unlink(self.temp_db.name)
    
    def test_regex_evaluation(self):
        """Test regex-based evaluation."""
        task = {
            "checker_type": "regex",
            "checker_config": {
                "pattern": r"The answer is (\d+)",
                "group": 0
            }
        }
        
        run_result = {
            "final_response": "The answer is 42",
            "status": "SUCCESS",
            "tokens_used": 100,
            "tool_calls_used": 0
        }
        
        # Create a mock run
        run_id = "test_run_1"
        self.data_store.conn.cursor().execute("""
            INSERT INTO runs (run_id, trace_id, genome_id, task_id, task_version, 
                            run_seed, status, start_timestamp, run_manifest_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, "trace_1", "genome_1", "task_1", 1, 12345, "SUCCESS", 
              "2026-01-01T00:00:00", "{}"))
        self.data_store.conn.commit()
        
        metrics = self.evaluator.evaluate(run_id, task, run_result)
        self.assertIn("pass_fail", metrics)
        self.assertEqual(metrics["pass_fail"], 1.0)


if __name__ == "__main__":
    unittest.main()
