"""
Metrics tracker for SpecDiff benchmark runs.
Collects throughput, latency, and acceptance rate per inference call.
"""


class EngineMetrics:
    """Stores and computes all performance metrics for a single inference run."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tokens = 0
        self.ttft = 0.0          # Time To First Token (ms)
        self.start_time = 0.0
        self.draft_accepted = 0
        self.draft_total = 0
        self.total_time = 0.0
        self.parity_verified = False
        self.perplexity = 0.0

    def get_throughput(self) -> float:
        """Overall throughput: total tokens / total wall-clock time (tok/s)."""
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time

    def get_avg_itl(self) -> float:
        """
        Average Inter-Token Latency (ms).
        Computed as: (total_time_ms - TTFT) / (total_tokens - 1).
        """
        if self.total_tokens <= 1:
            return 0.0
        return (self.total_time * 1000 - self.ttft) / (self.total_tokens - 1)

    def get_decode_throughput(self) -> float:
        """
        Decode-phase throughput, excluding TTFT warmup (tok/s).
        Reflects steady-state generation speed more accurately than raw throughput.
        """
        if self.total_tokens <= 1:
            return 0.0
        time_after_first = self.total_time - (self.ttft / 1000.0)
        if time_after_first <= 0:
            return 0.0
        return (self.total_tokens - 1) / time_after_first

    def get_acceptance_rate(self) -> float:
        """Draft acceptance rate α: accepted / total draft tokens proposed."""
        if self.draft_total == 0:
            return 0.0
        return self.draft_accepted / self.draft_total

    def to_dict(self) -> dict:
        """Serialize metrics to a flat dictionary for CSV/JSON export."""
        return {
            "throughput_tok_sec": self.get_throughput(),
            "decode_throughput_tok_sec": self.get_decode_throughput(),
            "ttft_ms": self.ttft,
            "avg_itl_ms": self.get_avg_itl(),
            "acceptance_rate_percent": self.get_acceptance_rate() * 100,
            "total_time_sec": self.total_time,
            "total_tokens": self.total_tokens,
            "parity_verified": self.parity_verified,
            "perplexity": self.perplexity,
        }
