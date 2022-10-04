from loguru import logger


def enforce(expression: bool, error_msg: str, exception_type=RuntimeError) -> None:
    if not expression:
        logger.error(error_msg)
        raise exception_type(error_msg)
