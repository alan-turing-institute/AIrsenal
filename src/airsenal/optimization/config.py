"""
Configuration for the optimisation.

These settings were previously repeated as default arguments across several
functions and again in the CLI, which is how the squad builder and the transfer
optimiser came to score benches differently without anyone deciding they should.
"""

from dataclasses import dataclass, field, replace

# The shape as_dict() produces and the scoring code consumes: a weight for the
# substitute goalkeeper, and one per outfield bench position.
SubWeightsDict = dict[str, float | tuple[float, float, float]]


@dataclass(frozen=True)
class SubWeights:
    """
    How much a substitute's predicted points count towards a squad's score.

    Outfield weights are ordered by bench position: first substitute, second, third.
    """

    gk: float = 0.03
    outfield: tuple[float, float, float] = (0.65, 0.3, 0.1)

    @classmethod
    def none(cls) -> "SubWeights":
        """Ignore the bench entirely."""
        return cls(gk=0.0, outfield=(0.0, 0.0, 0.0))

    def as_dict(self) -> SubWeightsDict:
        """The shape the scoring code still expects."""
        return {"GK": self.gk, "Outfield": self.outfield}


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Settings for the DEAP genetic algorithm used to pick a whole squad."""

    population_size: int = 100
    generations: int = 100
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    crossover_indpb: float = 0.5
    mutation_indpb: float = 0.1
    tournament_size: int = 3
    random_state: int | None = None
    verbose: bool = False

    def scaled(self, num_iterations: int) -> "GeneticAlgorithmConfig":
        """
        Population and generations both set from one number.

        Used by the wildcard and free-hit transfer strategies, which have a single
        num_iterations knob. Questionable - the two control different things - but
        it is what the code has always done, and it is at least explicit here.
        """
        return replace(self, population_size=num_iterations, generations=num_iterations)


@dataclass(frozen=True)
class SquadScoringConfig:
    """How a squad is scored during optimisation."""

    sub_weights: SubWeights = field(default_factory=SubWeights)
    dummy_sub_cost: int = 45
    budget: int = 1000


# Derived from SubWeights so there is one definition. The squad builder used to
# hard-code {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)} instead, so `optimize
# squad` and `optimize transfers` scored benches differently - unintentionally,
# and the docstrings advertised the other set again.
DEFAULT_SUB_WEIGHTS = SubWeights().as_dict()
