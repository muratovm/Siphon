class Folder():
    def __init__(self,name):
        self.name = name
        self.contents = []

    def add(self,item):
        self.contents.append(item)

class File():
    def __init__(self,name):
        self.name = name

def create_data_map():
    data_source = Folder("data")

    training = Folder("training")
    testing = Folder("testing")
    
    train_images = Folder("images")
    train_labels = Folder("labels")
    
    test_images = Folder("images")
    test_labels = Folder("labels")

    train_image_file = File("train-images-idx3-ubyte.gz")
    train_label_file = File("train-labels-idx1-ubyte.gz")
    test_image_file = File("t10k-images-idx3-ubyte.gz")
    test_label_file = File("t10k-labels-idx1-ubyte.gz")

    train_images.add(train_image_file)
    train_labels.add(train_label_file)
    test_images.add(test_image_file)
    test_labels.add(test_label_file)


    training.add(train_images)
    training.add(train_labels)

    testing.add(test_images)
    testing.add(test_labels)

    
    data_source.add(training)
    data_source.add(testing)

    return data_source

def build_data_map(data_source, data_structure, counts):
    depth = len(counts)
    for i in range(len(counts)-1):
        if counts[i] == -1:
            data_structure.write("\t"*(depth-1))
        else:
            data_structure.write("|\t"*(depth-1))
    if depth > 0 :
        data_structure.write("|---")
    data_structure.write(data_source.name+"\n")
                                 
    for count, element in enumerate(data_source.contents):
        if type(element) == Folder:
            build_data_map(element, data_structure, counts + [count-len(data_source.contents)])
        elif type(element) == File:
            for i in range(len(counts)):
                if counts[i] == -1:
                    data_structure.write("\t"*(depth-1))
                else:
                    data_structure.write("|\t"*(depth-1))
            if depth > 0 :
                data_structure.write("|---")
            data_structure.write(element.name+"\n")


data_source = create_data_map()
with open("data_structure.txt", "w") as data_structure:
    build_data_map(data_source,data_structure, [])

    
