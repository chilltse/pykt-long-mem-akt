"""Errors shared by the Python and HTTP interfaces."""


class ServiceError(ValueError):
    def __init__(self, code, message, status=422):
        super().__init__(message)
        self.code = code
        self.status = status
