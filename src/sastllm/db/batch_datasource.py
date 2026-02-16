from typing import Any, Callable, Iterable


class BatchDataSource:
    """
    Wraps a generator-producing function.
    Calling .iter() returns a fresh generator each time.
    """

    def __init__(self, generator_fn: Callable[[], Iterable[Any]]):
        self.generator_fn = generator_fn

    def iter(self) -> Iterable[Any]:
        return self.generator_fn()  # return NEW generator each call
