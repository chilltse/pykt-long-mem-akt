"""Simulation statistics, separate from observed labels and model probabilities."""
import math


def statistics(values):
    ordered = sorted(values)
    mean = sum(values) / len(values)
    def quantile(q):
        pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return dict(mean=mean, std=math.sqrt(sum((v-mean)**2 for v in values) / len(values)),
                q05=quantile(.05), q50=quantile(.5), q95=quantile(.95))
