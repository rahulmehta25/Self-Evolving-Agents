"""
Statistics Module: Bootstrap confidence intervals and significance testing.

Implements statistical methods for rigorous evaluation reporting.
"""

import random
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy import stats


def calculate_bootstrap_ci(
    data: List[float],
    metric_func: Callable[[List[float]], float] = np.mean,
    num_samples: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        data: List of data points
        metric_func: Function to compute the metric (default: mean)
        num_samples: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound, point_estimate)
    """
    if not data:
        return (0.0, 0.0, 0.0)
    
    sample_metrics = []
    n = len(data)
    
    for _ in range(num_samples):
        # Resample with replacement
        bootstrap_sample = random.choices(data, k=n)
        # Calculate metric on resample
        sample_metric = metric_func(bootstrap_sample)
        sample_metrics.append(sample_metric)
    
    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(sample_metrics, lower_percentile)
    upper_bound = np.percentile(sample_metrics, upper_percentile)
    point_estimate = metric_func(data)
    
    return (lower_bound, upper_bound, point_estimate)


def permutation_test(
    group1: List[float],
    group2: List[float],
    metric_func: Callable[[List[float]], float] = np.mean,
    num_permutations: int = 10000
) -> Tuple[float, float]:
    """
    Perform a permutation test to compare two groups.
    
    Args:
        group1: First group of data points
        group2: Second group of data points
        metric_func: Function to compute the metric (default: mean)
        num_permutations: Number of permutations to perform
    
    Returns:
        Tuple of (p_value, observed_difference)
    """
    if not group1 or not group2:
        return (1.0, 0.0)
    
    # Calculate observed difference
    observed_diff = metric_func(group1) - metric_func(group2)
    
    # Combine groups
    combined = group1 + group2
    n1 = len(group1)
    
    # Perform permutations
    extreme_count = 0
    
    for _ in range(num_permutations):
        # Randomly shuffle
        shuffled = combined.copy()
        random.shuffle(shuffled)
        
        # Split into two groups
        perm_group1 = shuffled[:n1]
        perm_group2 = shuffled[n1:]
        
        # Calculate difference
        perm_diff = metric_func(perm_group1) - metric_func(perm_group2)
        
        # Count if as or more extreme
        if abs(perm_diff) >= abs(observed_diff):
            extreme_count += 1
    
    p_value = extreme_count / num_permutations
    return (p_value, observed_diff)


def bootstrap_hypothesis_test(
    group1: List[float],
    group2: List[float],
    metric_func: Callable[[List[float]], float] = np.mean,
    num_samples: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, bool]:
    """
    Perform a bootstrap hypothesis test.
    
    Args:
        group1: First group of data points
        group2: Second group of data points
        metric_func: Function to compute the metric (default: mean)
        num_samples: Number of bootstrap samples
        confidence_level: Confidence level
    
    Returns:
        Tuple of (p_value, is_significant)
    """
    if not group1 or not group2:
        return (1.0, False)
    
    # Calculate observed difference
    observed_diff = metric_func(group1) - metric_func(group2)
    
    # Bootstrap the difference distribution
    differences = []
    n1, n2 = len(group1), len(group2)
    
    for _ in range(num_samples):
        # Resample each group with replacement
        boot1 = random.choices(group1, k=n1)
        boot2 = random.choices(group2, k=n2)
        
        # Calculate difference
        diff = metric_func(boot1) - metric_func(boot2)
        differences.append(diff)
    
    # Calculate p-value (two-tailed)
    if observed_diff >= 0:
        p_value = sum(1 for d in differences if d <= -abs(observed_diff)) / num_samples
    else:
        p_value = sum(1 for d in differences if d >= abs(observed_diff)) / num_samples
    
    p_value *= 2  # Two-tailed test
    is_significant = p_value < (1 - confidence_level)
    
    return (p_value, is_significant)


def win_tie_loss_analysis(
    group1: List[float],
    group2: List[float],
    significance_threshold: float = 0.05
) -> Dict[str, float]:
    """
    Perform win/tie/loss analysis for pairwise comparison.
    
    Args:
        group1: First group of data points
        group2: Second group of data points
        significance_threshold: P-value threshold for significance
    
    Returns:
        Dictionary with win_rate, tie_rate, loss_rate
    """
    if len(group1) != len(group2):
        # Paired comparison requires same length
        return {"win_rate": 0.0, "tie_rate": 0.0, "loss_rate": 0.0}
    
    wins = 0
    ties = 0
    losses = 0
    
    for g1, g2 in zip(group1, group2):
        diff = g1 - g2
        if abs(diff) < 0.01:  # Essentially equal
            ties += 1
        elif diff > 0:
            wins += 1
        else:
            losses += 1
    
    total = len(group1)
    return {
        "win_rate": wins / total if total > 0 else 0.0,
        "tie_rate": ties / total if total > 0 else 0.0,
        "loss_rate": losses / total if total > 0 else 0.0
    }


def calculate_aggregate_metrics(
    metrics_list: List[Dict[str, float]],
    metric_name: str
) -> Dict[str, float]:
    """
    Calculate aggregate statistics for a metric across multiple runs.
    
    Args:
        metrics_list: List of metric dictionaries
        metric_name: Name of the metric to aggregate
    
    Returns:
        Dictionary with mean, std, ci_lower, ci_upper
    """
    values = [m.get(metric_name, 0.0) for m in metrics_list if metric_name in m]
    
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "count": 0
        }
    
    mean = np.mean(values)
    std = np.std(values)
    ci_lower, ci_upper, _ = calculate_bootstrap_ci(values)
    
    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "count": len(values)
    }
