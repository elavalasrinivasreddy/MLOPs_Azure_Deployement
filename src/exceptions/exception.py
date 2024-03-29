import sys
from src.logger.logger import logging

class CustomException(Exception):

    def __init__(self,
                error_msg:str,
                error_details:sys):
        self.error_msg = error_msg
        _,_,exc_tb = error_details.exc_info()

        self.line = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        logging.error(f"Error occured in {self.filename} at line {self.line} \n error message: << {self.error_msg} >>")
        return f"Error occured in {self.filename} at line {self.line} \n error message: << {self.error_msg} >>"

if __name__=="__main__":

    try:
        a = 1/0
    except Exception as e:
        raise CustomException(e, sys)