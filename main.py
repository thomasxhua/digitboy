from file_reader import FileReader

# Within this repository it is located inside the assets folder:
PATH_TRAIN_IMAGES   = "assets/train-images-idx3-ubyte"
PATH_TRAIN_LABELS   = "assets/train-labels-idx1-ubyte"
PATH_TEST_IMAGES    = "assets/t10k-images-idx3-ubyte"
PATH_TEST_LABELS    = "assets/t10k-labels-idx1-ubyte"

def main():
    # Load data sets.
    # images  = FileReader(PATH_TRAIN_IMAGES, "img")
    # labels  = FileReader(PATH_TRAIN_LABELS, "lbl")
    images2 = FileReader(PATH_TEST_IMAGES, "img")
    labels2 = FileReader(PATH_TEST_LABELS, "lbl")

if __name__ == "__main__":
    main()