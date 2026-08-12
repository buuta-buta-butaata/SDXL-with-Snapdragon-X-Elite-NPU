import logging

RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

color_dict = {
    logging.FATAL: RED,
    logging.ERROR: RED,
    logging.WARNING: YELLOW,
    logging.DEBUG: BLUE,
}

def console_colored_filter(record):
    if record.levelno == logging.INFO:
        return True
    else:
        if record.levelno in color_dict.keys():
            color = color_dict[record.levelno]
            record.levelname = f"{color}{record.levelname}{END}"
    return True


if __name__ == "__main__":
    for i in range(10):
        for j in range(10):
            v = i * 10 + j
            print("\033[{}m{}\033[0m ".format(str(v), str(v).zfill(3)), end="")
        print("")

