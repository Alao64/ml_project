import sys
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_output = "Error occurred in script name [{0}] at line number [{1}] with error message [{2}]".format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )

    return error_output


class CustomException(Exception):
    def __init__(self, error_output, error_detail: sys):
        super().__init__(error_output)
        self.error_message = error_message_detail(error_output, error_detail)

    def __str__(self):
        return self.error_message
