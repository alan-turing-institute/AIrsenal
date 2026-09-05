"""What a failed call to an external service raises."""


class RemoteError(RuntimeError):
    """Something went wrong talking to an external service."""


class RemoteConnectionError(RemoteError):
    """The service could not be reached at all."""


class RemoteHTTPError(RemoteError):
    """The service answered, with a status that says no."""

    def __init__(self, msg: str, status_code: int | None = None) -> None:
        super().__init__(msg)
        self.status_code = status_code
