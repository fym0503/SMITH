"""Compatibility import for the manuscript-scale probe runners.

The implementation lives with the other Agent examples; this module preserves
the historical ``scripts.run_top128_feasibility_filtering`` import path used by
the batch ODT/OligoMiner/ProbeDealer runners.
"""

from scripts.agent_examples.run_top128_feasibility_filtering import *  # noqa: F401,F403
