"""
Stepwise model selection for ordination.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from .base import OrdinationResult


def ordistep(ordination: OrdinationResult, 
             env: Union[np.ndarray, pd.DataFrame],
             direction: str = "both",
             **kwargs) -> dict:
    """
    Stepwise model selection for ordination.
    
    Parameters:
        ordination: OrdinationResult object  
        env: Environmental data matrix
        direction: Direction of selection ("forward", "backward", "both")
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with selection results
    """
    # Placeholder implementation
    return {
        'selected_variables': [],
        'aic': 0.0,
        'direction': direction
    }