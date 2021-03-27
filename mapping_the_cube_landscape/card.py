import requests
import json
import string


def fetch_cards(scryfall_url, replace_json=None, lower=False):
    '''Fetches the default cards from Scryfall, renames to match CT's naming if given a replacement json'''
    
    if replace_json:
        rename_dict = json.load(open(replace_json))
    else:
        rename_dict = {}

    resp = requests.get(scryfall_url)
    json_data = resp.json()

    magic_cards = {}
    for card_data in json_data:
        
        if lower:
            name = card_data['name'].lower()
        else:
            name = card_data['name']
        
        magic_cards[name] = card_data

        # if a transform/flip card, only take front half name
        if card_data['layout'] in ['flip', 'transform', 'modal_dfc']:
            left_name, right_name = name.split(' // ')
            for name in [left_name, right_name]:
                magic_cards[name] = card_data['card_faces'][0]
                magic_cards[name]['cmc'] = card_data['cmc']
                magic_cards[name]['color_identity'] = card_data['color_identity']
            
                # rename if appropriate. We try to handle both cases here to prevent having to be case
                # specific in the rename json
                if name in rename_dict:
                    ct_spelling = rename_dict[name]
                    magic_cards[ct_spelling] = magic_cards[name]

        # rename some cards
        if name in rename_dict or name.lower() in rename_dict:
            ct_spelling = rename_dict[name.lower()]
            
            if not lower:
                ct_spelling = string.capwords(ct_spelling.lower())

            magic_cards[ct_spelling] = magic_cards[name]

    return magic_cards


def extract_types(name, type_line):
    '''Extracts all the card type given a type line'''

    possible_card_types = ['Creature', 'Artifact', 'Land', 'Planeswalker',
                           'Enchantment', 'Instant', 'Sorcery']
    types = [card_type for card_type in possible_card_types if card_type in type_line]

    # determine basic or nonbasic
    if 'Land' in type_line:
        if name in ['Mountain', 'Forest', 'Island', 'Plains', 'Swamp']:
            types.append('Basic Land')
        else:
            types.append('Nonbasic Land')

    return types


class Card:

    def __init__(self, name):

        self.name = name

    def extract_characteristics(self, card_dict):
        '''Uses a scryfall card dict object to define card characteristics'''

        card_data = card_dict[self.name]

        self.cmc = card_data.get('cmc')
        self.mana_cost = card_data.get('mana_cost')
        self.types = extract_types(self.name, card_data.get('type_line', ''))
        self.color_identity = card_data.get('color_identity')
