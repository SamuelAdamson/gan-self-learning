import os
import binascii

# DATA PREPROCESSING
# * Read in input file into bytes
# * Normalize the length of each file by padding with zero-bytes
# * Convert to hexadecimal
# * Convert hexadecimal digits to vector [0,1] 
#     i.e. 0 -> 0, 1 -> 1/16, 2 -> 2/16, ... , f -> 1


VECTOR_DICT = {
    "0": 0.0,
    "1": 0.0625,
    "2": 0.125,
    "3": 0.1875,
    "4": 0.25,
    "5": 0.3125,
    "6": 0.375,
    "7": 0.4375,
    "8": 0.5,
    "9": 0.5625,
    "A": 0.625,
    "B": 0.6875,
    "C": 0.75,
    "D": 0.8125,
    "E": 0.875,
    "F": 0.9375,
}


def preprocess_from_directory(directory_path: str) -> [[float]]:
    """ Preprocess folder of input files
     
    directory_path : path to input folder

    returns : vector data
    raises : FileNotFoundError, NotADirectoryError
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}!")

    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file[-4:].upper() == "PCAP"]
    max_len = max([os.path.getsize(fp) for fp in file_paths])

    result = [preprocess(fp, max_len) for fp in file_paths]
    return result


def preprocess(file_path: str, max_len: int=1024) -> [float]:
    """ Preprocess an input file (.pcap) 
    
    file_path : input file path
    max_len : maximum length of file in bytes

    returns : vector data
    raises : FileNotFoundError
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Directory not found: {file_path}")
    
    size = min(os.path.getsize(file_path), max_len)
    with open(file_path, "rb") as f:
        bytes = zero_pad(f.read())

    hex = binascii.hexlify(bytes)


    return []



def zero_pad(data: bytes, size: int, max_len: int) -> bytes:
    """ Pad bytes with zeroes """



if __name__ == "__main__":
    preprocess_from_directory("../ModbusTCP")