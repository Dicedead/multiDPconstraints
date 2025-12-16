from imports import *

Array = List[float] | np.ndarray
Float = float

DEFAULT_DOMAIN_START = 0.0
DEFAULT_DOMAIN_END = 1.0

TradeOffFunction = Callable[[Array], Array]