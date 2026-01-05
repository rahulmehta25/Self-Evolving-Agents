"""
Tests for statistics module.
"""

import unittest

from evoagentbench.evaluation.stats_module import (
    calculate_bootstrap_ci,
    permutation_test,
    win_tie_loss_analysis
)


class TestStatsModule(unittest.TestCase):
    """Test cases for statistics module."""
    
    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper, mean = calculate_bootstrap_ci(data, num_samples=100)
        
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
        self.assertAlmostEqual(mean, 3.0, places=1)
    
    def test_permutation_test(self):
        """Test permutation test."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        p_value, diff = permutation_test(group1, group2, num_permutations=100)
        
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        self.assertIsInstance(diff, float)
    
    def test_win_tie_loss(self):
        """Test win/tie/loss analysis."""
        group1 = [1.0, 2.0, 3.0]
        group2 = [2.0, 2.0, 4.0]
        
        result = win_tie_loss_analysis(group1, group2)
        
        self.assertIn("win_rate", result)
        self.assertIn("tie_rate", result)
        self.assertIn("loss_rate", result)


if __name__ == "__main__":
    unittest.main()
