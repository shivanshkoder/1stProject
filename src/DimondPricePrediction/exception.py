import sys

class ExceptionHandler(Exception):
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

    def __str__(self):
        return "Error occcured in python script name [{0}] at line number [{1}] with error message [{2}]".format(exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno, self.error_message)