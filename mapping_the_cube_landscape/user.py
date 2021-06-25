from collections import defaultdict
import pandas as pd
import numpy as np


class User:

    '''A class to store data about cubetutor users'''

    def __init__(self, cubeid):

        # attributes related to the cube
        self.cubeid = cubeid
        self.shortid = None
        self.user = None
        self.cube_name = None
        self.cube_size = None
        self.cube_category = None
        self.cube_list = None
        self.created_date = None
        self.updated_date = None
        self.percentage_fixing = None
        self.cube_image = None
        self.num_followers = None
        self.islisted = None
