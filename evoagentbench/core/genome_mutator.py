"""
Genome Mutator: Implements mutation operators for evolutionary optimization.

Provides mutation operators for modifying agent genomes.
"""

import copy
import random
from typing import Any, Dict, List, Optional


class GenomeMutator:
    """
    Mutates agent genomes for evolutionary optimization.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize the genome mutator.
        
        Args:
            mutation_rate: Probability of mutating each mutable parameter
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, parent_genome: Dict[str, Any], generation: int) -> Dict[str, Any]:
        """
        Create a mutated copy of a parent genome.
        
        Args:
            parent_genome: Parent genome to mutate
            generation: New generation number
            
        Returns:
            New mutated genome with new genome_id
        """
        import uuid
        
        # Deep copy the parent
        new_genome = copy.deepcopy(parent_genome)
        
        # Assign new ID and set parent
        new_genome["genome_id"] = str(uuid.uuid4())
        new_genome["parent_id"] = parent_genome["genome_id"]
        new_genome["generation"] = generation
        
        # Apply mutations
        if random.random() < self.mutation_rate:
            self._mutate_llm_config(new_genome)
        
        if random.random() < self.mutation_rate:
            self._mutate_system_prompt(new_genome)
        
        if random.random() < self.mutation_rate:
            self._mutate_tools(new_genome)
        
        if random.random() < self.mutation_rate:
            self._mutate_retrieval_config(new_genome)
        
        return new_genome
    
    def _mutate_llm_config(self, genome: Dict[str, Any]):
        """Mutate LLM configuration parameters."""
        llm_config = genome.get("llm_config", {})
        
        # Mutate temperature
        if "temperature" in llm_config:
            current_temp = llm_config["temperature"]
            # Small random change
            mutation = random.uniform(-0.2, 0.2)
            new_temp = max(0.0, min(2.0, current_temp + mutation))
            llm_config["temperature"] = round(new_temp, 2)
        
        # Mutate top_p
        if "top_p" in llm_config:
            current_top_p = llm_config["top_p"]
            mutation = random.uniform(-0.1, 0.1)
            new_top_p = max(0.0, min(1.0, current_top_p + mutation))
            llm_config["top_p"] = round(new_top_p, 2)
        
        # Mutate max_tokens
        if "max_tokens" in llm_config:
            current_max = llm_config["max_tokens"]
            mutation = random.randint(-100, 100)
            new_max = max(100, min(4000, current_max + mutation))
            llm_config["max_tokens"] = new_max
    
    def _mutate_system_prompt(self, genome: Dict[str, Any]):
        """Mutate system prompt (simple version - can be enhanced with LLM in v2)."""
        system_prompt = genome.get("system_prompt", "")
        
        if not system_prompt:
            return
        
        # Simple mutations: add/remove/modify instructions
        mutations = [
            lambda p: p + "\n\nBe concise in your responses.",
            lambda p: p + "\n\nThink step by step.",
            lambda p: p.replace("You are", "You are an expert"),
            lambda p: p + "\n\nAlways cite your sources.",
        ]
        
        if random.random() < 0.5:
            mutation_func = random.choice(mutations)
            genome["system_prompt"] = mutation_func(system_prompt)
    
    def _mutate_tools(self, genome: Dict[str, Any]):
        """Mutate tool configuration."""
        tools = genome.get("tools", [])
        
        if not tools:
            return
        
        # Randomly add or remove a tool (if available)
        if random.random() < 0.3 and len(tools) > 1:
            # Remove a random tool
            tools.pop(random.randint(0, len(tools) - 1))
        # Note: Adding tools requires knowing available tools, which would be
        # handled by the orchestrator
    
    def _mutate_retrieval_config(self, genome: Dict[str, Any]):
        """Mutate retrieval configuration."""
        retrieval_config = genome.get("retrieval_config", {})
        
        if "top_k" in retrieval_config:
            current_top_k = retrieval_config["top_k"]
            mutation = random.randint(-2, 2)
            new_top_k = max(1, min(20, current_top_k + mutation))
            retrieval_config["top_k"] = new_top_k
        
        if "similarity_threshold" in retrieval_config:
            current_threshold = retrieval_config["similarity_threshold"]
            mutation = random.uniform(-0.1, 0.1)
            new_threshold = max(0.0, min(1.0, current_threshold + mutation))
            retrieval_config["similarity_threshold"] = round(new_threshold, 2)
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                  generation: int) -> Dict[str, Any]:
        """
        Create a new genome by crossing over two parents.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            generation: New generation number
            
        Returns:
            New genome created by crossover
        """
        import uuid
        
        # Start with copy of parent1
        new_genome = copy.deepcopy(parent1)
        new_genome["genome_id"] = str(uuid.uuid4())
        new_genome["parent_id"] = f"{parent1['genome_id']}+{parent2['genome_id']}"
        new_genome["generation"] = generation
        
        # Crossover LLM config
        if random.random() < 0.5:
            new_genome["llm_config"] = copy.deepcopy(parent2["llm_config"])
        
        # Crossover system prompt (take from parent2 with some probability)
        if random.random() < 0.3:
            new_genome["system_prompt"] = parent2.get("system_prompt", new_genome["system_prompt"])
        
        # Crossover tools (merge or take from one parent)
        if random.random() < 0.5:
            tools1 = parent1.get("tools", [])
            tools2 = parent2.get("tools", [])
            # Take union of tools
            all_tools = {tool.get("name"): tool for tool in tools1}
            for tool in tools2:
                all_tools[tool.get("name")] = tool
            new_genome["tools"] = list(all_tools.values())
        
        return new_genome
