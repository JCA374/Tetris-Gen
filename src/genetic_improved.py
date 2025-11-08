"""
Improved genetic algorithm with safeguards against bad patterns.

Improvements over genetic.py:
1. Better fitness function that rewards survival + lines
2. Weight sign constraints to prevent bad patterns
3. Biased initialization near optimal range
4. Pattern monitoring during evolution
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from agent import TetrisAgent
from heuristics import HEURISTIC_NAMES


class Genome:
    """
    Represents a single genome with constraint awareness.
    """

    def __init__(self, weights=None, genome_id=None, use_constraints=True):
        """
        Initialize a genome with optional constraints.

        Args:
            weights: Array of heuristic weights
            genome_id: Unique identifier
            use_constraints: Whether to enforce sign constraints
        """
        self.id = genome_id if genome_id is not None else random.randint(0, 1000000)
        self.use_constraints = use_constraints

        if weights is None:
            # Initialize with biased random weights near optimal range
            self.weights = self._initialize_weights()
        else:
            self.weights = np.array(weights)
            if use_constraints:
                self.weights = self._apply_constraints(self.weights)

        self.fitness = 0
        self.games_played = 0

    def _initialize_weights(self):
        """
        Initialize weights with bias toward optimal ranges.
        """
        # Optimal ranges from research
        optimal_ranges = {
            'aggregate_height': (-1.0, -0.2),
            'complete_lines': (0.3, 1.0),
            'holes': (-1.0, -0.1),
            'bumpiness': (-0.5, -0.05),
            'max_height': (-1.0, -0.2),
            'wells': (-0.5, -0.05)
        }

        weights = []
        for name in HEURISTIC_NAMES:
            low, high = optimal_ranges.get(name, (-1, 1))
            # Add some noise for exploration
            weight = np.random.uniform(low, high)
            weights.append(weight)

        return np.array(weights)

    def _apply_constraints(self, weights):
        """
        Apply sign constraints to prevent bad patterns.
        """
        constraints = {
            'aggregate_height': 'negative',
            'complete_lines': 'positive',
            'holes': 'negative',
            'bumpiness': 'negative',
            'max_height': 'negative',
            'wells': 'negative'
        }

        constrained = weights.copy()
        for i, name in enumerate(HEURISTIC_NAMES):
            constraint = constraints.get(name)
            if constraint == 'negative' and constrained[i] > 0:
                constrained[i] = -abs(constrained[i])
            elif constraint == 'positive' and constrained[i] < 0:
                constrained[i] = abs(constrained[i])

        return constrained

    def __repr__(self):
        return f"Genome(id={self.id}, fitness={self.fitness:.2f})"


class ImprovedGeneticAlgorithm:
    """
    Improved genetic algorithm with pattern monitoring.
    """

    def __init__(
        self,
        population_size=30,
        mutation_rate=0.1,
        mutation_step=0.2,
        elite_size=2,
        tournament_size=3,
        use_constraints=True,
        use_improved_fitness=True
    ):
        """
        Initialize the improved genetic algorithm.

        Args:
            population_size: Number of genomes in the population
            mutation_rate: Probability of mutating each gene
            mutation_step: Maximum relative change during mutation
            elite_size: Number of top performers to preserve
            tournament_size: Number of individuals in tournament selection
            use_constraints: Whether to enforce weight sign constraints
            use_improved_fitness: Use improved fitness function
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_step = mutation_step
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.use_constraints = use_constraints
        self.use_improved_fitness = use_improved_fitness

        self.population: List[Genome] = []
        self.generation = 0
        self.best_genome: Genome = None
        self.history: List[Dict] = []

    def initialize_population(self):
        """
        Create initial population with biased initialization.
        """
        self.population = [
            Genome(genome_id=i, use_constraints=self.use_constraints)
            for i in range(self.population_size)
        ]
        print(f"Initialized population of {self.population_size} genomes")
        if self.use_constraints:
            print("  ✓ Weight sign constraints enabled")
        if self.use_improved_fitness:
            print("  ✓ Improved fitness function enabled")

    def evaluate_fitness(self, env_creator, games_per_genome=3):
        """
        Evaluate fitness with improved scoring.
        """
        for i, genome in enumerate(self.population):
            total_score = 0
            total_lines = 0
            total_steps = 0

            for game_num in range(games_per_genome):
                env = env_creator()
                agent = TetrisAgent(genome.weights)

                result = agent.play_game(env, render=False)

                total_score += result['score']
                total_lines += result['lines']
                total_steps += result['steps']

                env.close()

            # Average across games
            avg_lines = total_lines / games_per_genome
            avg_score = total_score / games_per_genome
            avg_steps = total_steps / games_per_genome

            # Improved fitness function
            if self.use_improved_fitness:
                # Exponential reward for lines + survival bonus
                line_reward = (avg_lines ** 1.5) * 100
                survival_bonus = avg_steps * 0.5
                genome.fitness = line_reward + avg_score + survival_bonus
            else:
                # Original fitness
                genome.fitness = avg_lines * 100 + avg_score

            genome.games_played += games_per_genome

            print(f"  Genome {i+1}/{self.population_size}: "
                  f"Lines={avg_lines:.1f}, Score={avg_score:.1f}, "
                  f"Steps={avg_steps:.1f}, Fitness={genome.fitness:.1f}")

    def check_bad_patterns(self):
        """
        Check population for bad weight patterns.
        """
        warnings = []

        for genome in self.population:
            weights_dict = {name: w for name, w in zip(HEURISTIC_NAMES, genome.weights)}

            # Check for bad signs
            if weights_dict.get('complete_lines', 1) < 0:
                warnings.append(f"Genome {genome.id}: complete_lines is negative!")
            if weights_dict.get('holes', -1) > 0:
                warnings.append(f"Genome {genome.id}: holes is positive!")
            if weights_dict.get('aggregate_height', -1) > 0:
                warnings.append(f"Genome {genome.id}: aggregate_height is positive!")
            if weights_dict.get('wells', -1) > 0:
                warnings.append(f"Genome {genome.id}: wells is positive (center stacking risk)!")

        return warnings

    def selection(self) -> Genome:
        """Tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda g: g.fitness)

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Uniform crossover."""
        child_weights = np.zeros_like(parent1.weights)

        for i in range(len(child_weights)):
            if random.random() < 0.5:
                child_weights[i] = parent1.weights[i]
            else:
                child_weights[i] = parent2.weights[i]

        return Genome(weights=child_weights, use_constraints=self.use_constraints)

    def mutate(self, genome: Genome) -> Genome:
        """Mutate with constraints."""
        mutated_weights = genome.weights.copy()

        for i in range(len(mutated_weights)):
            if random.random() < self.mutation_rate:
                # Mutate by percentage
                change = mutated_weights[i] * self.mutation_step * random.uniform(-1, 1)
                mutated_weights[i] += change

                # Occasional random noise
                if random.random() < 0.1:
                    mutated_weights[i] += random.uniform(-0.5, 0.5)

        new_genome = Genome(weights=mutated_weights, use_constraints=self.use_constraints)
        return new_genome

    def evolve_population(self):
        """Evolve with pattern checking."""
        # Check for bad patterns
        warnings = self.check_bad_patterns()
        if warnings:
            print("\n⚠ PATTERN WARNINGS:")
            for w in warnings[:5]:  # Show first 5
                print(f"  {w}")

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Update best
        if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
            self.best_genome = Genome(weights=self.population[0].weights.copy(),
                                     use_constraints=self.use_constraints)
            self.best_genome.fitness = self.population[0].fitness

        # Record stats
        fitnesses = [g.fitness for g in self.population]
        stats = {
            'generation': self.generation,
            'best_fitness': fitnesses[0],
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': fitnesses[-1],
            'best_weights': self.population[0].weights.copy()
        }
        self.history.append(stats)

        print(f"\nGeneration {self.generation} Statistics:")
        print(f"  Best Fitness: {stats['best_fitness']:.2f}")
        print(f"  Avg Fitness: {stats['avg_fitness']:.2f}")
        print(f"  Worst Fitness: {stats['worst_fitness']:.2f}")

        # Create new population
        new_population = []

        # Elitism
        for i in range(self.elite_size):
            elite = Genome(weights=self.population[i].weights.copy(),
                          use_constraints=self.use_constraints)
            new_population.append(elite)

        # Breed rest
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def get_best_genome(self) -> Genome:
        """Get best genome."""
        if self.best_genome is None:
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            return self.population[0]
        return self.best_genome
