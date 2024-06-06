from box import Box
import sys

def printl(*args, log_path=None, **kwargs):
    """
    Enhanced print function that logs to a file if log_path is provided.

    Parameters:
    *args: Variable length argument list, passed to print.
    log_path (str, optional): The file path to log the output. If None, acts like the standard print.
    **kwargs: Arbitrary keyword arguments, passed to print.
    """
    # Perform the standard print operation
    print(*args, **kwargs)

    # If a log_path is provided, also write the output to the specified log file
    if log_path:
        # Capture the print output
        old_stdout = sys.stdout
        sys.stdout = open(log_path, 'a')
        print(*args, **kwargs)  # Output goes to the log file
        sys.stdout.close()
        sys.stdout = old_stdout


def load_configs(config_dict):
    configs = Box(config_dict)
    return configs