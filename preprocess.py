from random import shuffle, choice
import os
import logging
logger = logging.getLogger(__name__)

class Preprocess:

    def __init__(self, directory_path):
        """
        Return a randomized list of each directory's contents

        :param directory_path: a directory that contains sub-folders of images

        :returns class_files: a dict of each file in each folder
        """
        logger.debug('Initializing Preprocess')
        super().__init__()
        self.directory_path = directory_path
        self.class_files = self.__get_list()
        self.min_images = self.__count_min_number_of_images()
        self.set_list = self.__group_into_triples()

    def __get_list(self):
        logging.debug('Getting initial list of images')
        class_files = dict()
        classes = os.listdir(self.directory_path)

        for x in classes:
            class_files[x] = []
            for y in os.listdir(os.path.join(self.directory_path, x)):
                class_files[x].append(os.path.join(self.directory_path, x, y))

            i = class_files[x]
            shuffle(i)
            class_files[x] = i

        return class_files

    def __count_min_number_of_images(self):
        logger.debug('Counting the number of images')
        min_images = None
        for k, v in self.class_files.items():
            if min_images is None or min_images < len(v):
                min_images = len(v)
        return min_images

    def __count_number_of_groups(self):
        return self.class_files.items().__len__()

    def __get_indexes(self):
        """
        Construct a list of three indexes for anchor, pos, neg
        """
        n_groups = self.__count_number_of_groups()

        # Choose an anchor/positive index
        a_idx = choice(range(n_groups))
        p_idx = a_idx

        # Choose a negative sample
        n_idx = p_idx
        while n_idx == p_idx:
            n_idx = choice(range(n_groups))

        return a_idx, p_idx, n_idx

    def __get_dictname_from_indexes(self, a_idx, p_idx, n_idx):
        idx_names = list(self.class_files.keys())
        return idx_names[a_idx], idx_names[p_idx], idx_names[n_idx]

    def __group_into_triples(self):
        """ Convert sorted individual lists into triples
        """
        logger.debug('Grouping into triples')
        set_list = []
        for i in range(self.min_images):
            try:
                a_idx, p_idx, n_idx = self.__get_indexes()
                a_idx, p_idx, n_idx = self.__get_dictname_from_indexes(a_idx, p_idx, n_idx)

                # Only pop from the anchor, otherwise random sample from others
                a_img = self.class_files[a_idx].pop()
                p_img = choice(self.class_files[p_idx])
                n_img = choice(self.class_files[n_idx])
                l = (a_img, p_img, n_img)
                set_list.append(l)
            except IndexError:
                pass  # this is for when you run out of samples
        return set_list


