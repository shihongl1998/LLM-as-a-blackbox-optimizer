import os
import re

from engine.datasets.benchmark import Benchmark, read_split, split_trainval, save_split

OPENAI_UCF101_CLASSNAMES = [
    'Apply Eye Makeup',
    'Apply Lipstick',
    'Archery',
    'Baby Crawling',
    'Balance Beam',
    'Band Marching',
    'Baseball Pitch',
    'Basketball',
    'Basketball Dunk',
    'Bench Press',
    'Biking',
    'Billiards',
    'Blow Dry Hair',
    'Blowing Candles',
    'Body Weight Squats',
    'Bowling',
    'Boxing Punching Bag',
    'Boxing Speed Bag',
    'Breast Stroke',
    'Brushing Teeth',
    'Clean And Jerk',
    'Cliff Diving',
    'Cricket Bowling',
    'Cricket Shot',
    'Cutting In Kitchen',
    'Diving',
    'Drumming',
    'Fencing',
    'Field Hockey Penalty',
    'Floor Gymnastics',
    'Frisbee Catch',
    'Front Crawl',
    'Golf Swing',
    'Haircut',
    'Hammer Throw',
    'Hammering',
    'Hand Stand Pushups',
    'Handstand Walking',
    'Head Massage',
    'High Jump',
    'Horse Race',
    'Horse Riding',
    'Hula Hoop',
    'Ice Dancing',
    'Javelin Throw',
    'Juggling Balls',
    'Jump Rope',
    'Jumping Jack',
    'Kayaking',
    'Knitting',
    'Long Jump',
    'Lunges',
    'Military Parade',
    'Mixing',
    'Mopping Floor',
    'Nunchucks',
    'Parallel Bars',
    'Pizza Tossing',
    'Playing Cello',
    'Playing Daf',
    'Playing Dhol',
    'Playing Flute',
    'Playing Guitar',
    'Playing Piano',
    'Playing Sitar',
    'Playing Tabla',
    'Playing Violin',
    'Pole Vault',
    'Pommel Horse',
    'Pull Ups',
    'Punch',
    'Push Ups',
    'Rafting',
    'Rock Climbing Indoor',
    'Rope Climbing',
    'Rowing',
    'Salsa Spin',
    'Shaving Beard',
    'Shotput',
    'Skate Boarding',
    'Skiing',
    'Skijet',
    'Sky Diving',
    'Soccer Juggling',
    'Soccer Penalty',
    'Still Rings',
    'Sumo Wrestling',
    'Surfing',
    'Swing',
    'Table Tennis Shot',
    'Tai Chi',
    'Tennis Swing',
    'Throw Discus',
    'Trampoline Jumping',
    'Typing',
    'Uneven Bars',
    'Volleyball Spiking',
    'Walking With Dog',
    'Wall Pushups',
    'Writing On Board',
    'Yo Yo',
]

class UCF101(Benchmark):

    dataset_name = "ucf101"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")

        train, val, test = read_split(self.split_path, self.image_dir)
        ## Uncomment the following lines to generate a new split
        # cname2lab = {}
        # filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
        # with open(filepath, "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         label, classname = line.strip().split(" ")
        #         label = int(label) - 1  # conver to 0-based index
        #         cname2lab[classname] = label

        # trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
        # test = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
        # train, val = split_trainval(trainval)
        # save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)
        self.classnames = OPENAI_UCF101_CLASSNAMES
        self.lab2cname = {i: c for i, c in enumerate(self.classnames)}

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)
                item = {"impath": impath, "label": label, "classname": renamed_action}
                items.append(item)

        return items
