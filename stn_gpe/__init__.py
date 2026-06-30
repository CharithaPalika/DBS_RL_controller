# stn_gpe/__init__.py
"""
STN-GPe model package
Contains models, simulation scripts, and analysis utilities
for studying STN–GPe dynamics under various conditions.
"""

from .stngpe import STN_GPe_loop
from .dbs import GenerateDBS
from .analysis import Analysis
from .stn_gpe_weights import interconnectivity, wt_four_blocks, random_wts_sparse
from .utils import load_yaml, save_yaml, lfp_dist_matrix

# Define what should be accessible when importing `stn_gpe`
__all__ = [
    "STN_GPe_loop",
    "GenerateDBS",
    "Analysis",
    "interconnectivity",
    "wt_four_blocks",
    "random_wts_sparse",
    "load_yaml",
    "save_yaml",
    "lfp_dist_matrix",
]