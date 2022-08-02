import os
import numpy as np
from PIL import Image
import math
import struct


class FileReader:
    # Path constants.
    PATH_OUT_IMAGES     = "out/"
    BMP                 = ".bmp"

    # Information about the data sets from their website (http://yann.lecun.com/exdb/mnist/):
    """
    Images:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns

    Labels:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    """
    IMAGES_NUM_COLS       = 28
    IMAGES_NUM_ROWS       = 28

    # Magic Numbers to indentify files.
    MAGIC_NUMBER_LABELS   = 0x00000801 # 2049
    MAGIC_NUMBER_IMAGES   = 0x00000803 # 2051
    MAGIC_NUMBER_NETWORK  = 0x00000805 # 2053

    # Characters to draw 8 different shades of grey for displaying image.
    COLOR_STRING = " .-+*?$@"
    # COLOR_STRING = "@$?*+-. " # uncomment if you want "dark theme".

    # Clears all files in directory.
    @staticmethod
    def empty_dir(directory):
        for filename in os.listdir(directory):
            file = os.path.join(directory, filename)
            os.remove(file)
            print("Removed \"" + filename + "\".")

    # Creates string from image as 2d array.
    @staticmethod
    def image_to_string(image):
        image_string = ""
        image_width = int(math.sqrt(len(image)))
        for i in range(len(image)):
            # print(image[i])
            pixel           = int(image[i]) >> 5
            image_string    += FileReader.COLOR_STRING[pixel] + " "
            if (i % image_width == 0):
                image_string += "\n"
        return image_string

    # Shrinks 28x28 image array to 14x14 image array.
    @staticmethod
    def shrink_image(image):
        new_width   = int(FileReader.IMAGES_NUM_COLS) >> 1
        new_size    = new_width * new_width
        new_image   = np.empty(new_size)
        for i in range(new_width):
            for j in range(new_width):
                # Take every second index from old image.
                old_i = int(i) << 1
                old_j = int(j) << 1
                # Create sum for a 2x2 area in old image. To calculate average from sum, devide by
                # four (and floor), since four summands.
                new_image[i*new_width + j] = int(
                    image[old_i*FileReader.IMAGES_NUM_COLS + old_j]
                    + image[old_i*FileReader.IMAGES_NUM_COLS + old_j+1]
                    + image[(old_i+1)*FileReader.IMAGES_NUM_COLS + old_j]
                    + image[(old_i+1)*FileReader.IMAGES_NUM_COLS + old_j+1]
                ) >> 2
        return new_image


    # Initialize FileReader by saving file_name, opening the file, reading the magic number to
    # confirm a files idendity, reading number of items to start reading of data.
    def __init__(self, file_name, item_type):
        self.file_name          = file_name
        self.file               = open(file_name, "rb")
        self.magic_number       = self.read_magic_number()
        self.number_of_items    = self.read_number_of_items()
        self.items              = self.read_items(item_type)
        self.file.close()

    # Reads n bytes (MSB) and converts to int.
    def read_bytes_to_int(self, n):
        return int.from_bytes(self.file.read(n), "big")

    # Reads n bytes (MSB) and converts to float.
    def read_bytes_to_float(self, n):
        return struct.unpack('f', self.file.read(n))

    # Read magic number from file.
    def read_magic_number(self):
        # Reset "bookmark".
        self.file.seek(0, 0)
        read_number = self.read_bytes_to_int(4)
        # Check, which data set magic number corresponds to:
        print(
            "Successfully opened \"" + self.file_name
            + "\".\nGot magic number " + str(read_number) + ", ", end=""
        )
        # Exits, if no magic number belonging to any data set was detected.
        if (read_number == self.MAGIC_NUMBER_LABELS):
            print("label", end="")
        elif (read_number == self.MAGIC_NUMBER_IMAGES):
            print("image", end="")
        elif (read_number == self.MAGIC_NUMBER_WEIGHTS):
            print("weights", end="")
        elif (read_number == self.MAGIC_NUMBER_BIASES):
            print("biases", end="")
        else:
            print("doesn't seem to belong to any known format.")
            print("\"" + self.file_name + "\" incorrect? Exiting execution.")
            exit()
        print(" file detected. Continuing...")
        return read_number

    # Reads 4 bytes to determinte number of items in data set.
    def read_number_of_items(self):
        self.file.seek(4, 0)
        return self.read_bytes_to_int(4)

    # Reads 28x28 bytes from file and saves into 2d array.
    def read_image(self, n):
        image_size  = self.IMAGES_NUM_ROWS * self.IMAGES_NUM_COLS
        self.file.seek(n*image_size + 4*4, 0)
        image_array = np.empty(image_size)
        # Fill up image array.
        for i in range(len(image_array)):
            image_array[i] = self.read_bytes_to_int(1)
        return image_array

    # Reads byte from file and returns as integer.
    def read_label(self, n):
        self.file.seek(n + 2*4, 0)
        return self.read_bytes_to_int(1)

    # Read bytes from file and returns a float.
    def read_float(self, n):
        # Assuming Python uses float32, therefore 32bits = 4 byte.
        self.file.seek(n*4 + 2*4, 0)
        [x] = self.read_bytes_to_float(4)
        return x

    # Choose function to read item and returns item on n-th position.
    def read_item(self, n, item_type):
        if (item_type == "img"):
            return self.read_image(n)
        elif (item_type == "lbl"):
            return self.read_label(n)
        elif (item_type == "float"):
            return self.read_float(n)
        else:
            return self.read_bytes_to_int(n)

    # Read items from file.
    def read_items(self, item_type):
        # Reading bytes as array into list images.
        items = []
        print("Reading items from file... ")
        for i in range(self.number_of_items):
            items.append(self.read_item(i, item_type))
            if (i % 100 == 0):
                print("[" + str(i) + "/" + str(self.number_of_items) + "]", end="\r")
        print(
            "Completed reading file. " + str(self.number_of_items) + " items of type \""
            + item_type + "\" read.\n"
        )
        return items

    # Takes array and converts it to grayscale image as bmp.
    def create_images_from_array(self, images):
        FileReader.empty_dir(self.PATH_OUT_IMAGES)
        for i in range(len(images)):
            image       = Image.fromarray(images[i]).convert("L")
            image_name  = str(i) + self.BMP
            image.save(self.PATH_OUT_IMAGES + image_name)
            print("Created image \"" + image_name + "\".", end="\r")

    # Open file and wipe all its contents.
    def wipe(self):
        self.file.truncate(0)
