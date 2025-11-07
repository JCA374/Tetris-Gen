"""
Genetic algorithm for evolving Tetris-playing agents.

The algorithm maintains a population of weight vectors (genomes),
evaluates their fitness by playing Tetris, and evolves them through
selection, crossover, and mutation.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from agent import TetrisAgent
from heuristics import HEURISTIC_NAMES


class Genome:
    """
    Represents a single genome (individual) in the population.
    """

    def __init__(self, weights=None, genome_id=None):
        """
        Initialize a genome.

        Args:
            weights: Array of heuristic weights. If None, generates random weights.
            genome_id: Unique identifier for this genome
        """
        self.id = genome_id if genome_id is not None else random.randint(0, 1000000)

        if weights is None:
            # Initialize with random weights between -10 and 10
            self.weights = np.random.uniform(-10, 10, len(HEURISTIC_NAMES))
        else:
            self.weights = np.array(weights)

        self.fitness = 0
        self.games_played = 0

    def __repr__(self):
        return f"Genome(id={self.id}, fitness={self.fitness:.2f})"


class GeneticAlgorithm:
    """
    Genetic algorithm for evolving Tetris agents.
    """

    def __init__(
        self,
        population_size=30,
        mutation_rate=0.1,
        mutation_step=0.2,
        elite_size=1,
        tournament_size=3
    ):
        """
        Initialize the genetic algorithm.

        Args:
            population_size: Number of genomes in the population
            mutation_rate: Probability of mutating each gene
            mutation_step: Maximum relative change during mutation (as fraction)
            elite_size: Number of top performers to preserve unchanged
            tournament_size: Number of individuals in tournament selection
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_step = mutation_step
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        self.population: List[Genome] = []
        self.generation = 0
        self.best_genome: Genome = None
        self.history: List[Dict] = []

    def initialize_population(self):
        """
        Create the initial population with random genomes.
        """
        self.population = [
            Genome(genome_id=i) for i in range(self.population_size)
        ]
        print(f"Initialized population of {self.population_size} genomes")

    def evaluate_fitness(self, env_creator, games_per_genome=3):
        """
        Evaluate the fitness of all genomes by playing games.

        Args:
            env_creator: Function that creates a new gym environment
            games_per_genome: Number of games to play per genome for averaging
        """
        for i, genome in enumerate(self.population):
            total_score = 0
            total_lines = 0

            # Play multiple games and average the results
            for game_num in range(games_per_genome):
                env = env_creator()
                agent = TetrisAgent(genome.weights)

                result = agent.play_game(env, render=False)

                # Fitness is primarily based on lines cleared
                # with bonus for score
                total_score += result['score']
                total_lines += result['lines']

                env.close()

            # Average fitness across games
            avg_lines = total_lines / games_per_genome
            avg_score = total_score / games_per_genome

            # Fitness function: prioritize lines cleared, with score as tiebreaker
            genome.fitness = avg_lines * 100 + avg_score
            genome.games_played += games_per_genome

            print(f"  Genome {i+1}/{self.population_size}: "
                  f"Lines={avg_lines:.1f}, Score={avg_score:.1f}, "
                  f"Fitness={genome.fitness:.1f}")

    def selection(self) -> Genome:
        """
        Select a genome using tournament selection.

        Returns:
            Selected genome
        """
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda g: g.fitness)

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """
        Create offspring through uniform crossover.

        Args:
            parent1: First parent genome
            parent2: Second parent genome

        Returns:
            New offspring genome
        """
        child_weights = np.zeros_like(parent1.weights)

        for i in range(len(child_weights)):
            # Randomly inherit from either parent
            if random.random() < 0.5:
                child_weights[i] = parent1.weights[i]
            else:
                child_weights[i] = parent2.weights[i]

        return Genome(weights=child_weights)

    def mutate(self, genome: Genome) -> Genome:
        """
        Mutate a genome's weights.

        Args:
            genome: Genome to mutate (creates a copy)

        Returns:
            Mutated genome
        """
        mutated_weights = genome.weights.copy()

        for i in range(len(mutated_weights)):
            if random.random() < self.mutation_rate:
                # Mutate by adding/subtracting a percentage of current value
                change = mutated_weights[i] * self.mutation_step * random.uniform(-1, 1)
                mutated_weights[i] += change

                # Also occasionally add random noise for exploration
                if random.random() < 0.1:
                    mutated_weights[i] += random.uniform(-2, 2)

        return Genome(weights=mutated_weights)

    def evolve_population(self):
        """
        Create the next generation through selection, crossover, and mutation.
        """
        # Sort population by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Track best genome
        if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
            self.best_genome = Genome(weights=self.population[0].weights.copy())
            self.best_genome.fitness = self.population[0].fitness

        # Record statistics
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

        # Elitism: Keep the best performers
        for i in range(self.elite_size):
            elite = Genome(weights=self.population[i].weights.copy())
            new_population.append(elite)

        # Generate rest of population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def get_best_genome(self) -> Genome:
        """
        Get the best genome found so far.

        Returns:
            Best genome
        """
        if self.best_genome is None:
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            return self.population[0]
        return self.best_genome

    def save_checkpoint(self, filepath):
        """
        Save the current state of the genetic algorithm.

        Args:
            filepath: Path to save the checkpoint
        """
        import pickle

        checkpoint = {
            'generation': self.generation,
            'population': [(g.weights, g.fitness) for g in self.population],
            'best_genome': (self.best_genome.weights, self.best_genome.fitness) if self.best_genome else None,
            'history': self.history,
            'config': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'mutation_step': self.mutation_step,
                'elite_size': self.elite_size,
                'tournament_size': self.tournament_size
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load a saved checkpoint.

        Args:
            filepath: Path to the checkpoint file
        """
        import pickle

        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        self.generation = checkpoint['generation']
        self.history = checkpoint['history']

        # Restore population
        self.population = [
            Genome(weights=weights) for weights, fitness in checkpoint['population']
        ]
        for genome, (_, fitness) in zip(self.population, checkpoint['population']):
            genome.fitness = fitness

        # Restore best genome
        if checkpoint['best_genome']:
            weights, fitness = checkpoint['best_genome']
            self.best_genome = Genome(weights=weights)
            self.best_genome.fitness = fitness

        print(f"Checkpoint loaded from {filepath}")
        print(f"Resumed at generation {self.generation}")
