import os
import datetime
import card
import re
import numpy as np


def flatten_decklist(decklist):
    '''Takes in a list of "num card" and returns a flattened version
       with just card names'''

    flattened_deck = []
    for line in decklist:
        num = line.split(' ')[0]
        card = ' '.join(line.split(' ')[1:])
        flattened_deck.extend([card]*(int(num)))

    return flattened_deck


class Deck:

    '''A class for storing deck characteristics'''

    def __init__(self):

        self.date = None
        self.deck_id = None
        self.main = None
        self.side = None
        self.built = None
        self.draft = None
        self.colors = None

    def parse_text_decklist(self, path):
        '''Parses the decklist if passed as a text file,
        returns main, sideboard as lists'''

        # if the user passes a path, do some minor parsing to start
        date_string = os.path.basename(path).split('_')[1]
        self.date = datetime.datetime.strptime(date_string, "%Y-%m-%d-%H-%M")
        self.deck_id = os.path.basename(path).split('_')[2][:-4]

        # read in deck, skip title
        with open(path, 'r') as deck:
            deck = [line.strip('\n') for line in deck.readlines()][1:]

        # every deck has sideboard listed
        sideboard_index = deck.index('Sideboard')
        main = deck[:sideboard_index - 1]
        side = deck[sideboard_index + 1:]

        # flatten the decklist
        self.main = flatten_decklist(main)
        self.side = flatten_decklist(side)

        nonbasics = set(self.main + self.side) - \
            set(['Mountain', 'Island', 'Forest', 'Swamp', 'Plains'])
        # determine whether the decklist is built or not
        if len(self.main) > 20 and len(self.main) < 43:
            self.built = True
        else:
            self.built = False

        self.draft = True if len(nonbasics) < 50 else False

        return None

    def determine_colors(self, magic_cards):
        '''Determine the major colors of the deck based on nonland cards in the main.
        Requires a dictionary of magic cards for lookup (see card.py)'''

        nonland_colors = [''.join(magic_cards[card]['color_identity'])
                          for card in self.main if 'Land' not in magic_cards[card]['type_line']]
        deck_colors = list(''.join(nonland_colors))
        deck_colors, counts = np.unique(deck_colors, return_counts=True)
        color_dict = dict(zip(deck_colors, counts))
        self.colors = color_dict

        return color_dict
