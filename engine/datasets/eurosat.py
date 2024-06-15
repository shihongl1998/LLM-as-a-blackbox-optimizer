import os
import pickle

from engine.datasets.benchmark import Benchmark, read_split, read_and_split_data, save_split

NEW_CNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}

OPENAI_EUROSAT_CLASSNAMES = ['annual crop land', 'forest', 'brushland or shrubland', 'highway or road', 'industrial buildings or commercial buildings', 'pasture land', 'permanent crop land', 'residential buildings or homes or apartments', 'river', 'lake or sea']

class EuroSAT(Benchmark):

    dataset_name = "eurosat"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")

        train, val, test = read_split(self.split_path, self.image_dir)

        # # Uncomment the following lines to generate a new split
        # train, val, test = read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
        # save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)
        self.classnames = OPENAI_EUROSAT_CLASSNAMES # This overwrites the classnames with OpenAI's version
        self.lab2cname = {i: c for i, c in enumerate(self.classnames)}