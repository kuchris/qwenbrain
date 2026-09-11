"""Response-only presence penalty for Transformers generation."""

from transformers import LogitsProcessor


class PresencePenalty(LogitsProcessor):
    """Subtract a fixed penalty once for each token present in this response.

    Prompt/history tokens are excluded. Repeated occurrences of a generated token
    do not compound the penalty; this is not a frequency penalty or a hard ban.
    """

    def __init__(self, penalty, prompt_length):
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if not generated.numel():
            return scores
        penalized = scores.gather(1, generated) - self.penalty
        return scores.scatter(1, generated, penalized)
