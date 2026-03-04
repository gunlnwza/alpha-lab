from dataclasses import dataclass

from alpha_lab.utils import ForexData


@dataclass
class Bar:
    idx: int
    open: float
    high: float
    low: float
    close: float


class _SafeArrayView:
    def __init__(self, array, parent):
        self._array = array
        self._parent = parent

    def __getitem__(self, idx):
        now = self._parent.now

        if isinstance(idx, int):
            if idx > now or idx < 0:
                raise IndexError("Lookahead detected")
        elif isinstance(idx, slice):
            if idx.stop is not None and idx.stop > now + 1:
                raise IndexError("Lookahead detected")

        return self._array[idx]


class PrecomputedData:
    def __init__(self, forex_data: ForexData):
        object.__setattr__(self, "_forex_data", forex_data)
        object.__setattr__(self, "_i", 0)
        object.__setattr__(self, "_custom", {})

    @property
    def now(self):
        return self._i
    
    @property
    def bar(self) -> Bar:
        i = self._i
        return Bar(
            i,
            self._forex_data.open[i],
            self._forex_data.high[i],
            self._forex_data.low[i],
            self._forex_data.close[i],
        )
    
    @property
    def forex_data(self) -> ForexData:
        return self._forex_data

    def step(self):
        object.__setattr__(self, "_i", self._i + 1)

    def is_last_bar(self):
        return self._i >= len(self._forex_data) - 1

    def __setattr__(self, name, value):
        if name.startswith("_"):  # internal attributes
            object.__setattr__(self, name, value)
        else:  # store user feature arrays
            self._custom[name] = value

    def __getattr__(self, name):  # Called only if normal attribute not found
        if name in self._custom:
            arr = self._custom[name]
            return _SafeArrayView(arr, self)

        raise AttributeError(f"{name} not found")
