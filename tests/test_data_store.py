"""
Tests for DataStore.
"""

import os
import tempfile
import unittest

from evoagentbench.core.data_store import DataStore


class TestDataStore(unittest.TestCase):
    """Test cases for DataStore."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.data_store = DataStore(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.data_store.close()
        os.unlink(self.temp_db.name)
    
    def test_save_and_get_genome(self):
        """Test saving and retrieving genomes."""
        genome = {
            "genome_id": "test_genome_1",
            "generation": 0,
            "system_prompt": "Test prompt",
            "llm_config": {"model_name": "test", "temperature": 0.0}
        }
        
        genome_id = self.data_store.save_genome(genome)
        self.assertEqual(genome_id, "test_genome_1")
        
        retrieved = self.data_store.get_genome(genome_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["genome_id"], "test_genome_1")
    
    def test_save_and_get_task(self):
        """Test saving and retrieving tasks."""
        task = {
            "task_id": "test_task_1",
            "version": 1,
            "category": ["test"],
            "difficulty": "easy",
            "checker_type": "regex",
            "prompt_template": "Test task",
            "gold_answer": {},
            "checker_config": {},
            "budget": {"max_tokens": 100, "max_tool_calls": 5, "max_time_seconds": 30}
        }
        
        self.data_store.save_task(task)
        retrieved = self.data_store.get_task("test_task_1", 1)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["task_id"], "test_task_1")


if __name__ == "__main__":
    unittest.main()
