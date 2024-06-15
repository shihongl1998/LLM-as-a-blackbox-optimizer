import os

from engine.datasets.benchmark import Benchmark, read_split, read_and_split_data, save_split

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}

# The only difference from OPENAI's version is that we only have one face class
OPENAI_CALTECH101_CLASSES = ['face', 'leopard', 'motorbike', 'accordion', 'airplane', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera',
     'cannon', 'side of a car', 'ceiling fan', 'cellphone', 'chair', 'chandelier', 'body of a cougar cat', 'face of a cougar cat', 'crab', 'crayfish', 'crocodile', 'head of a  crocodile',
     'cup', 'dalmatian', 'dollar bill', 'dolphin', 'dragonfly', 'electric guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'head of a flamingo', 'garfield',
     'gerenuk', 'gramophone', 'grand piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline skate', 'joshua tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama',
     'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver',
     'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea horse', 'snoopy (cartoon beagle)', 'soccer ball', 'stapler', 'starfish', 'stegosaurus', 'stop sign',
     'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water lilly', 'wheelchair', 'wild cat', 'windsor chair', 'wrench', 'yin and yang symbol']

class Caltech101(Benchmark):

    dataset_name = "caltech-101"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        train, val, test = read_split(self.split_path, self.image_dir)
        
        # # Uncomment the following lines to generate a new split
        # train, val, test = read_and_split_data(
        #     self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
        # save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)
        self.classnames = OPENAI_CALTECH101_CLASSES # This overwrites the classnames with OpenAI's version
        self.lab2cname = {i: c for i, c in enumerate(self.classnames)}
     
