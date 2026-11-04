"""Same complete density/flow unit, with all original success demonstrations."""

from .distribution_data import DistributionTrainingData


class DenseDistributionTrainingData(DistributionTrainingData):
    def __init__(self, training_support, demonstrations, normalizer, *, include_policy=True, policy_cache=None):
        # The established __getitem__ performs the unchanged original-target,
        # timestamp, normalizer and whole-source-episode retrieval checks.
        # DenseTrainingSupport delegates query() to the separate reference index.
        self.library = training_support
        self.demonstrations = demonstrations
        self.normalizer = normalizer
        self.include_policy = include_policy
        self.policy_cache = policy_cache
