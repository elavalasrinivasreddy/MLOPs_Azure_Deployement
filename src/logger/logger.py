import logging
import os
from datetime import datetime

log_file_name = f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_file_path = os.path.join(os.getcwd(),"logs")
os.makedirs(log_file_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_file_path,log_file_name),
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )

logging.info("Log is initiated..!")