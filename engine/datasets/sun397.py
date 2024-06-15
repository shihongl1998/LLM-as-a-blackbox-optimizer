import os

from engine.datasets.benchmark import Benchmark, read_split, save_split, split_trainval

OPENAI_SUN397_CLASSNAMES = [
    'abbey', 'airplane cabin', 'airport terminal', 'alley', 'amphitheater', 'amusement arcade', 'amusement park', 'anechoic chamber', 'apartment building outdoor', 'apse indoor', 'aquarium', 'aqueduct',
    'arch', 'archive', 'arrival gate outdoor', 'art gallery', 'art school', 'art studio', 'assembly line', 'athletic field outdoor', 'atrium public', 'attic', 'auditorium', 'auto factory', 'badlands',
    'badminton court indoor', 'baggage claim', 'bakery shop', 'balcony exterior', 'balcony interior', 'ball pit', 'ballroom', 'bamboo forest', 'banquet hall', 'bar', 'barn', 'barndoor', 'baseball field',
    'basement', 'basilica', 'basketball court outdoor', 'bathroom', 'batters box', 'bayou', 'bazaar indoor', 'bazaar outdoor', 'beach', 'beauty salon', 'bedroom', 'berth', 'biology laboratory', 'bistro indoor',
    'boardwalk', 'boat deck', 'boathouse', 'bookstore', 'booth indoor', 'botanical garden', 'bow window indoor', 'bow window outdoor', 'bowling alley', 'boxing ring', 'brewery indoor', 'bridge', 'building facade',
    'bullring', 'burial chamber', 'bus interior', 'butchers shop', 'butte', 'cabin outdoor', 'cafeteria', 'campsite', 'campus', 'canal natural', 'canal urban', 'candy store', 'canyon',
    'car interior backseat', 'car interior frontseat', 'carrousel', 'casino indoor', 'castle', 'catacomb', 'cathedral indoor', 'cathedral outdoor', 'cavern indoor', 'cemetery', 'chalet', 'cheese factory',
    'chemistry lab', 'chicken coop indoor', 'chicken coop outdoor', 'childs room', 'church indoor', 'church outdoor', 'classroom', 'clean room', 'cliff', 'cloister indoor', 'closet', 'clothing store', 'coast',
    'cockpit', 'coffee shop', 'computer room', 'conference center', 'conference room', 'construction site', 'control room', 'control tower outdoor', 'corn field', 'corral', 'corridor', 'cottage garden',
    'courthouse', 'courtroom', 'courtyard', 'covered bridge exterior', 'creek', 'crevasse', 'crosswalk', 'cubicle office', 'dam', 'delicatessen', 'dentists office', 'desert sand', 'desert vegetation', 'diner indoor',
    'diner outdoor', 'dinette home', 'dinette vehicle', 'dining car', 'dining room', 'discotheque', 'dock', 'doorway outdoor', 'dorm room', 'driveway', 'driving range outdoor', 'drugstore', 'electrical substation',
    'elevator door', 'elevator interior', 'elevator shaft', 'engine room', 'escalator indoor', 'excavation', 'factory indoor', 'fairway', 'fastfood restaurant', 'field cultivated', 'field wild', 'fire escape',
    'fire station', 'firing range indoor', 'fishpond', 'florist shop indoor', 'food court', 'forest broadleaf', 'forest needleleaf', 'forest path', 'forest road', 'formal garden', 'fountain', 'galley', 'game room',
    'garage indoor', 'garbage dump', 'gas station', 'gazebo exterior', 'general store indoor', 'general store outdoor', 'gift shop', 'golf course', 'greenhouse indoor', 'greenhouse outdoor', 'gymnasium indoor',
    'hangar indoor', 'hangar outdoor', 'harbor', 'hayfield', 'heliport', 'herb garden', 'highway', 'hill', 'home office', 'hospital', 'hospital room', 'hot spring', 'hot tub outdoor', 'hotel outdoor', 'hotel room',
    'house', 'hunting lodge outdoor', 'ice cream parlor', 'ice floe', 'ice shelf', 'ice skating rink indoor', 'ice skating rink outdoor', 'iceberg', 'igloo', 'industrial area', 'inn outdoor', 'islet', 'jacuzzi indoor',
    'jail cell', 'jail indoor', 'jewelry shop', 'kasbah', 'kennel indoor', 'kennel outdoor', 'kindergarden classroom', 'kitchen', 'kitchenette', 'labyrinth outdoor', 'lake natural', 'landfill', 'landing deck',
    'laundromat', 'lecture room', 'library indoor', 'library outdoor', 'lido deck outdoor', 'lift bridge', 'lighthouse', 'limousine interior', 'living room', 'lobby', 'lock chamber', 'locker room', 'mansion',
    'manufactured home', 'market indoor', 'market outdoor', 'marsh', 'martial arts gym', 'mausoleum', 'medina', 'moat water', 'monastery outdoor', 'mosque indoor', 'mosque outdoor', 'motel', 'mountain',
    'mountain snowy', 'movie theater indoor', 'museum indoor', 'music store', 'music studio', 'nuclear power plant outdoor', 'nursery', 'oast house', 'observatory outdoor', 'ocean', 'office', 'office building',
    'oil refinery outdoor', 'oilrig', 'operating room', 'orchard', 'outhouse outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking garage indoor', 'parking garage outdoor', 'parking lot', 'parlor', 'pasture',
    'patio', 'pavilion', 'pharmacy', 'phone booth', 'physics laboratory', 'picnic area', 'pilothouse indoor', 'planetarium outdoor', 'playground', 'playroom', 'plaza', 'podium indoor', 'podium outdoor', 'pond',
    'poolroom establishment', 'poolroom home', 'power plant outdoor', 'promenade deck', 'pub indoor', 'pulpit', 'putting green', 'racecourse', 'raceway', 'raft', 'railroad track', 'rainforest', 'reception',
    'recreation room', 'residential neighborhood', 'restaurant', 'restaurant kitchen', 'restaurant patio', 'rice paddy', 'riding arena', 'river', 'rock arch', 'rope bridge', 'ruin', 'runway', 'sandbar',
    'sandbox', 'sauna', 'schoolhouse', 'sea cliff', 'server room', 'shed', 'shoe shop', 'shopfront', 'shopping mall indoor', 'shower', 'skatepark', 'ski lodge', 'ski resort', 'ski slope', 'sky', 'skyscraper',
    'slum', 'snowfield', 'squash court', 'stable', 'stadium baseball', 'stadium football', 'stage indoor', 'staircase', 'street', 'subway interior', 'subway station platform', 'supermarket', 'sushi bar', 'swamp',
    'swimming pool indoor', 'swimming pool outdoor', 'synagogue indoor', 
    'synagogue outdoor', 'television studio', 'temple east asia', 'temple south asia', 'tennis court indoor', 'tennis court outdoor', 'tent outdoor', 'theater indoor procenium', 'theater indoor seats',
    'thriftshop', 'throne room', 'ticket booth', 'toll plaza', 'topiary garden', 'tower', 'toyshop', 'track outdoor', 'train railway', 'train station platform', 'tree farm', 
    'treehouse', 'trench', 'underwater coral reef', 'utility room', 'valley', 'van interior', 'vegetable garden', 'veranda', 'veterinarians office', 'viaduct', 'videostore', 'village', 'vineyard',
    'volcano', 'volleyball court indoor', 'volleyball court outdoor', 'waiting room', 'warehouse indoor', 'water tower', 'waterfall block', 'waterfall fan', 'waterfall plunge', 'watering hole', 'wave', 'wet bar', 'wheat field', 'wind farm', 'windmill', 'wine cellar barrel storage', 'wine cellar bottle storage', 'wrestling ring indoor', 'yard', 'youth hostel']

class SUN397(Benchmark):

    dataset_name = "sun397"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")

        train, val, test = read_split(self.split_path, self.image_dir)
        # classnames = []
        # with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip()[1:]  # remove /
        #         classnames.append(line)
        # cname2lab = {c: i for i, c in enumerate(classnames)}
        # trainval = self.read_data(cname2lab, "Training_01.txt")
        # test = self.read_data(cname2lab, "Testing_01.txt")
        # train, val = split_trainval(trainval)
        # save_split(train, val, test, self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)
        self.classnames = OPENAI_SUN397_CLASSNAMES
        self.lab2cname = {i: c for i, c in enumerate(self.classnames)}

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)
                item = {'impath': impath, 'label': label, 'classname': classname}
                items.append(item)

        return items
