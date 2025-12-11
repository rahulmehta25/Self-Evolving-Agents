"""
Evolution Engine: Implements the core evolutionary loop.

Handles selection, mutation, and crossover to generate new generations.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from .data_store import DataStore
from .genome_mutator import GenomeMutator


class SelectionStrategy:
    """Base class for selection strategies."""
    
    def select_parents(self, fitness_scores: List[Tuple[str, float]], 
                      num_parents: int) -> List[str]:
        """
        Select parent genomes based on fitness.
        
        Args:
            fitness_scores: List of (genome_id, fitness) tuples
            num_parents: Number of parents to select
            
        Returns:
            List of selected genome IDs
        """
        raise NotImplementedError


class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy."""
    
    def __init__(self, tournament_size: int = 5):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of genomes to compete in each tournament
        """
        self.tournament_size = tournament_size
    
    def select_parents(self, fitness_scores: List[Tuple[str, float]], 
                      num_parents: int) -> List[str]:
        """
        Select parents using tournament selection.
        
        Args:
            fitness_scores: List of (genome_id, fitness) tuples
            num_parents: Number of parents to select
            
        Returns:
            List of selected genome IDs
        """
        if not fitness_scores:
            return []
        
        parents = []
        for _ in range(num_parents):
            # Randomly select tournament participants
            tournament = random.sample(fitness_scores, 
                                     min(self.tournament_size, len(fitness_scores)))
            # Select the best from tournament
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        
        return parents


class ElitistSelection(SelectionStrategy):
    """Elitist selection - always keeps top performers."""
    
    def __init__(self, elite_fraction: float = 0.05):
        """
        Initialize elitist selection.
        
        Args:
            elite_fraction: Fraction of top genomes to keep
        """
        self.elite_fraction = elite_fraction
    
    def select_elite(self, fitness_scores: List[Tuple[str, float]]) -> List[str]:
        """
        Select elite genomes to carry over.
        
        Args:
            fitness_scores: List of (genome_id, fitness) tuples
            
        Returns:
            List of elite genome IDs
        """
        if not fitness_scores:
            return []
        
        # Sort by fitness (descending)
        sorted_scores = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
        
        # Select top fraction
        num_elite = max(1, int(len(sorted_scores) * self.elite_fraction))
        elite = [genome_id for genome_id, _ in sorted_scores[:num_elite]]
        
        return elite


class EvolutionEngine:
    """
    Evolution engine that manages the evolutionary loop.
    """
    
    def __init__(self, data_store: DataStore,
                 selection_strategy: Optional[SelectionStrategy] = None,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.2,
                 elite_fraction: float = 0.05):
        """
        Initialize the evolution engine.
        
        Args:
            data_store: DataStore instance
            selection_strategy: Selection strategy (default: TournamentSelection)
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_fraction: Fraction of elite genomes to keep
        """
        self.data_store = data_store
        self.selection_strategy = selection_strategy or TournamentSelection()
        self.elitist_selection = ElitistSelection(elite_fraction)
        self.mutator = GenomeMutator(mutation_rate)
        self.crossover_rate = crossover_rate
    
    def evolve_generation(self, current_generation: int,
                         population_size: int,
                         fitness_metric: str = "weighted_fitness") -> List[str]:
        """
        Evolve a new generation from the current generation.
        
        Args:
            current_generation: Current generation number
            population_size: Desired population size for new generation
            fitness_metric: Metric name to use for fitness
            
        Returns:
            List of new genome IDs
        """
        # 1. Get fitness scores for current generation
        fitness_scores = self.data_store.get_fitness_by_generation(
            current_generation, fitness_metric
        )
        
        if not fitness_scores:
            raise ValueError(f"No fitness scores found for generation {current_generation}")
        
        # 2. Select elite genomes to carry over
        elite_genomes = self.elitist_selection.select_elite(fitness_scores)
        num_elite = len(elite_genomes)
        num_new = population_size - num_elite
        
        # 3. Generate new genomes
        new_genome_ids = []
        new_generation = current_generation + 1
        
        # Carry over elite (without mutation)
        for elite_id in elite_genomes:
            elite_genome = self.data_store.get_genome(elite_id)
            if elite_genome:
                # Create copy with new generation number
                import copy
                import uuid
                new_genome = copy.deepcopy(elite_genome)
                new_genome["genome_id"] = str(uuid.uuid4())
                new_genome["parent_id"] = elite_id
                new_genome["generation"] = new_generation
                new_id = self.data_store.save_genome(new_genome)
                new_genome_ids.append(new_id)
        
        # Generate remaining genomes through selection, mutation, and crossover
        while len(new_genome_ids) < population_size:
            if random.random() < self.crossover_rate and len(fitness_scores) >= 2:
                # Crossover
                parents = self.selection_strategy.select_parents(fitness_scores, 2)
                if len(parents) == 2:
                    parent1 = self.data_store.get_genome(parents[0])
                    parent2 = self.data_store.get_genome(parents[1])
                    if parent1 and parent2:
                        child = self.mutator.crossover(parent1, parent2, new_generation)
                        new_id = self.data_store.save_genome(child)
                        new_genome_ids.append(new_id)
                        continue
            
            # Mutation
            parents = self.selection_strategy.select_parents(fitness_scores, 1)
            if parents:
                parent = self.data_store.get_genome(parents[0])
                if parent:
                    child = self.mutator.mutate(parent, new_generation)
                    new_id = self.data_store.save_genome(child)
                    new_genome_ids.append(new_id)
        
        return new_genome_ids[:population_size]
    
    def get_best_genome(self, generation: int,
                       fitness_metric: str = "weighted_fitness") -> Optional[str]:
        """
        Get the best genome from a generation.
        
        Args:
            generation: Generation number
            fitness_metric: Metric name to use for fitness
            
        Returns:
            Genome ID of the best genome, or None
        """
        fitness_scores = self.data_store.get_fitness_by_generation(
            generation, fitness_metric
        )
        
        if not fitness_scores:
            return None
        
        # Return genome with highest fitness
        best_genome_id, _ = max(fitness_scores, key=lambda x: x[1])
        return best_genome_id
