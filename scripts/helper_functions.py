import logging

# Configure logging to write to a file
logging.basicConfig(
    filename='logfile_training.txt',  # Specify the file name
    level=logging.DEBUG,      # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date/time format
)



