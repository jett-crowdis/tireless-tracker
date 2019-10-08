import argparse
import requests
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import datetime
from matplotlib import rc
import matplotlib.pyplot as plt

def fetch_cards():
    '''Fetches the default cards from Scryfall'''

    print('Fetching cards from Scryfall...')
    resp = requests.get('https://archive.scryfall.com/json/scryfall-default-cards.json')
    json_data = resp.json()
    magic_cards = {}
    for i in range(len(json_data)):
        card_name = json_data[i]['name']
        if card_name not in magic_cards.keys():
            magic_cards[card_name] = {'color': ''.join(json_data[i]['color_identity']),
                                      'cmc': json_data[i]['cmc'],
                                      'type': json_data[i]['type_line']}

    
        if json_data[i]['layout'] == 'transform': magic_cards[card_name.split('//')[0].strip(' ')] = magic_cards[card_name]

    return magic_cards

def make_deck(infile):

    '''Given a deck text file, analyze its contents. Outputs the main/sideboard rate of cards, the win/loss, deck colors, archetypes'''

    maindeck, side = [], []

    with open(infile) as deck_file:

        # extract deck "meta-data" - the colors, archetypes, and game records. Does not currently do anything with match records
        summary = [line.strip('\n') for line in deck_file.readlines()]
        deck_color = summary[0].split(':')[1].strip(' ')
        deck_archetypes = summary[1].split(':')[1].strip(' ').split('_')
        deck_record = list(map(float,summary[3].split(':')[1].strip(' ').split('-')))

        # extract the cards in the decklist - will extract sideboard as well.
        cards = []
        for card_info in summary[5:]:
            card_info = card_info.strip('\n')
            if card_info == '':
                cards.extend([card_info])
                continue
            else:
                num, card = card_info.split(' ')[0],' '.join(card_info.split(' ')[1:])
                cards.extend([card]*int(num))
        try:
            div = cards.index('')
            maindeck, side = cards[:div], cards[div+1:]
        except:
            maindeck = cards
            side = []
        win, loss = deck_record

    return maindeck, side, deck_color, deck_archetypes, win, loss

def extract_decklists(directory, magic_cards, date_arg):

    '''Parses all the decklists in a directory and creates a dictionary to contain this info.'''
    misspellings = open('misspellings.txt', 'w')
    misspellings.write('The following cards are not found in Scryfall\'s database:\n')
    deck_dict = {}

    # extract and make the decklist for every deck file in the input directory
    for i, infile in enumerate(os.listdir(directory)):

        if infile[-4:] != '.txt': continue # added to avoid '.DS_store', etc

        # attempt to analyze decklist. If unable, skip it. Will extract date if it exists.
        try:
            maindeck, side, color, archetypes, win, loss = make_deck(os.path.join(directory,infile))
            
            if date_arg:
                date = infile.split('_')[-1][:-4]
        except:
            print('File {} could not be analyzed.'.format(infile))
            continue

        for card in maindeck + side: 
            if not magic_cards.get(card): 
                misspellings.write('{} in file {}\n'.format(card, infile))

        deck_dict[i] = {'main': maindeck, 'side': side, 'color': color, 'archetypes': archetypes, 'record':[win, loss]}
        if date_arg: deck_dict[i]['date'] = date

    return deck_dict

def find_card_type(full_type):
    
    '''Takes in a card_type (str) and returns its shortened type (Artifact, Creature, Enchantment, PW, Land, Sorcery, Instant)'''

    for card_type in ['Creature', 'Artifact', 'Enchantment', 'Planeswalker', 'Land', 'Sorcery', 'Instant']:
        if card_type in full_type:
            return card_type

    return None

def export_card_analysis(deck_list_dict, magic_cards, card_filter, archetype_dict):
    
    '''Analyzes card representation and win rates and exports them to csv. If the normalize argument is true, it normalizes 
    card win rates to the deck win rates.'''
    
    card_dict = {}

    # loop through dictionary, extract info on individual cards
    for deck_dict in deck_list_dict.values():

        # extract deck, get its record
        deck = deck_dict['main']
        win, loss = map(int, deck_dict['record'])

        # get the deck archetype for normalization
        archetypes = deck_dict['archetypes']
        if len(archetypes) == 1: archetype = 'Pure ' + archetypes[0]
        else: archetype = archetypes[-1]

        # loop through cards in deck, check if they exist in Scryfall. If they do, store game information.
        for card in deck:
            
            if not magic_cards.get(card): continue

            if card not in card_dict.keys():
                card_dict[card] = {'win': win, 'loss': loss, 'num': 1, 'archetypes': [archetype]}
            else:
                card_dict[card]['num'] += 1
                card_dict[card]['win'] += win
                card_dict[card]['loss'] += loss
                card_dict[card]['archetypes'] += [archetype]

    print('{} unique cards identified in decklists'.format(len(card_dict)))

    # extract information about the cards from scryfall dictionary
    for card in card_dict.keys():
        color, cmc, card_type = magic_cards[card].values()

        for characteristic, value in zip(['color', 'cmc', 'type'], [color, cmc, find_card_type(card_type)]):
            card_dict[card][characteristic] = value

        # get win rates and perform normalization by archetype win rate
        card_dict[card]['win %'] = card_dict[card]['win']/(card_dict[card]['win'] + card_dict[card]['loss'])
        archetype_winrates = [archetype_dict[archetype]['Win %'] for archetype in card_dict[card]['archetypes']]
        card_dict[card]['win %/arch %'] = card_dict[card]['win %']/np.average(archetype_winrates)

    # store the results of the card analysis in a dataframe. Then calculate win %, apply filter, and export to csv.
    results = {card: {key: card_dict[card][key] for key in ['win', 'loss', 'num', 'color', 'cmc', 'type', 'win %', 'win %/arch %']} for card in card_dict.keys()}
    results_df = pd.DataFrame.from_dict(results, orient = 'index').reset_index()
    results_df.columns = ['Name','Win', 'Loss','Num','Color', 'CMC', 'Type', 'Win %', 'Win %/Arch %']

    if card_filter:
        results_df = results_df.loc[results_df['Num'] > card_filter]

    win_percent_sorted = results_df.sort_values(by = 'Win %', ascending = False)
    norm_percent_sorted = results_df.sort_values(by = 'Win %/Arch %', ascending = False)
    win_percent_sorted.to_csv('Card_Analysis_Win%.csv', index = False)
    norm_percent_sorted.to_csv('Card_Analysis_Norm%.csv', index = False)

    return card_dict

def export_archetype_analysis(deck_list_dict):
    
    '''Analyzes archetype distribution and exports to csv. Will analyze by subtypes as well.'''
    
    archetype_dict = defaultdict(lambda: {'num':0, 'win': 0, 'loss': 0})

    archetypes = []

    # loop through each deck, and store archetype information
    for deck_dict in deck_list_dict.values():
        
        wins, losses = map(int, deck_dict['record'])

        archetypes.extend(deck_dict['archetypes'])    
        
        if len(deck_dict['archetypes']) == 1:
            archetype = deck_dict['archetypes'][0]
            archetype_dict['Pure ' + archetype]['num'] += 1
            archetype_dict['Pure ' + archetype]['win'] += wins
            archetype_dict['Pure ' + archetype]['loss'] += losses

        for archetype in deck_dict['archetypes']:

            archetype_dict[archetype]['num'] += 1
            archetype_dict[archetype]['win'] += wins
            archetype_dict[archetype]['loss'] += losses

    # calculate win rates for each archetype
    for archetype in archetype_dict:
        archetype_dict[archetype]['Win %'] = archetype_dict[archetype]['win']/(archetype_dict[archetype]['win'] + archetype_dict[archetype]['loss'])
    
    # convert archetype analysis to dataframe, calculate winrates, then export.
    archetype_df = pd.DataFrame.from_dict(archetype_dict, orient = 'index').reset_index()
    archetype_df.columns = ['Archetype','Num','Win', 'Loss', 'Win %']
    archetype_df.to_csv('Archetype_Analysis.csv', index = False)

    return archetype_dict


def export_color_analysis(deck_dict, magic_cards):

    '''Analyze color distribution in cards and decks and exports to csv'''

    deck_colors, card_colors = [], defaultdict(lambda: [])
    
    # loop through decklists, and for each extract the deck colors and the colors of the nonland cards
    for deck in deck_dict.values():
        
        main_colors = [magic_cards[card]['color'] for card in deck['main'] if magic_cards.get(card) and find_card_type(magic_cards[card]['type']) != 'Land']
        deck_colors.extend(list(deck['color']))
        colors, counts = np.unique(main_colors, return_counts = True)
        counts = counts/sum(counts)
        color_dict = dict(zip(colors, counts))

        # only consider a color when it is more than 15% of a deck's composition (prevents splashed outliers from shifting the analysis)
        for color, count in color_dict.items():
            if count > 0.15: card_colors[color].append(count)

    # calculate the average for deck colors and card colors
    deck_color, deck_num = np.unique(deck_colors, return_counts = True)
    card_colors = {color:np.average(counts) for color, counts in card_colors.items()}

    # export analysis
    color_df = pd.DataFrame.from_dict(dict(zip(deck_color, deck_num/sum(deck_num))), orient = 'index').reset_index()
    color_df.columns = ['Color','Deck_Spread']
    color_df['Card_Spread'] = [card_colors[color] for color in deck_color]
    color_df.to_csv('Color_Analysis.csv', index = False)

def export_timecourse_analysis(deck_dict, window):

    '''If specified, analyze the decklists and the archetype win rates over time. Returns a dataframe that is then plotted.'''

    # extract all the archetypes present in the decklists, and the corresponding dates
    decklists = deck_dict.values()
    archetypes = [deck['archetypes'] for deck in decklists]
    archetypes = list(set([archetype for archetype_list in archetypes for archetype in archetype_list]))
    dates = [int(deck['date']) for deck in decklists]

    # sort the decklists based on the date they were added.
    sorted_decklists = [deck for _, deck in sorted(zip(dates,decklists))]
    window_num = len(sorted_decklists) - window + 1
    storage_matrix = np.zeros([len(archetypes), window_num])

    # conduct a sliding window analysis, storing the average win rate for the archetypes during this window.
    for i in range(window_num):
        decklist_window = sorted_decklists[i:i+window]
        for j, archetype in enumerate(archetypes):
            records = np.array([deck['record'] for deck in decklist_window if archetype in deck['archetypes']])
            storage_matrix[j, i] = np.sum(records, axis = 0)[0]/np.sum(records)

    return archetypes, storage_matrix

def plot_timecourse(archetypes, storage_matrix):

    '''Given a dataframe containing time course information, plot the win rates'''
    fig, ax = plt.subplots(1,1, figsize = (10,6))

    # only does this for the super archetypes (Aggro, Midrange, Control). Other archetypes are too infrequent to get a good picture.
    colors = {'Aggro':'#ffa600', 'Midrange':'#bc5090', 'Control':'#003f5c'}
    for i, archetype in enumerate(archetypes):
        if archetype in ['Reanimator', 'Combo', 'Ramp']: continue
        data = storage_matrix[i, :]
        ax.plot(range(len(data)), data, label = '{}'.format(archetype), color = colors[archetype])

    ax.legend(fontsize = 14)
    plt.xlabel('Decks', fontsize = 18)
    plt.ylabel('Rolling Average', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.title('Archetype Winrates', fontsize = 18)
    plt.tight_layout()
    fig.savefig('Archetype_Winrates.png', dpi = 300)

def main():
    parser = argparse.ArgumentParser(description='Analyzes decklists given an input folder')
    parser.add_argument('-d','--deck_folder', type=str, metavar='\b', help = 'folder containing decklist text files', required = True)
    parser.add_argument('-f','--filter', type=int, metavar='\b', help = 'cards with frequency below this filter will be excluded'
                        , default = 0)
    parser.add_argument('-date','--date', type=bool, metavar='\b', help = 'if your decklist names have date information, will perform timecourse analysis of archetypes'
                        , default = False)
    parser.add_argument('-w', '--window', type=int, metavar = '\b', help = 'size of sliding window in time course analysis', default = 100)
        
    args = parser.parse_args()

    decklist_folder = args.deck_folder
    card_filter = args.filter
    date_arg = args.date
    window = args.window

    # get cards from scryfall
    magic_cards = fetch_cards()

    # extract decklists 
    deck_dict = extract_decklists(decklist_folder, magic_cards, date_arg)
    print('{} decks extracted.'.format(len(deck_dict)))

    # export the card, archetype, and color analysis
    print('Analyzing archetypes, cards, and colors...')
    archetype_dict = export_archetype_analysis(deck_dict)
    export_card_analysis(deck_dict, magic_cards, card_filter, archetype_dict)
    export_color_analysis(deck_dict, magic_cards)
    
    # if specified, output time course analysis too.
    if date_arg: 
        print('Analyzing time course...')
        archetypes, timecourse = export_timecourse_analysis(deck_dict, window)
        plot_timecourse(archetypes, timecourse)

if __name__ == '__main__':
    main()