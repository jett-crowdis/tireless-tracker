import numpy as np
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import defaultdict
import argparse
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def simulate_match(deck0, deck1):
    '''inputs: deck0, deck1 - array of length 23 corresponding to deck strength
       output: win, loss - integers correponding to the game wins and losses of deck0'''
    
    p = sum(deck0)/(sum(deck0) + sum(deck1)) # probability that deck1 beats deck2 in a game
    games = np.random.choice([0,1], p = [1-p, p], size = 3) # plays three games
    
    if sum(games[:2]) == 0: 
        win, loss = 0, 2 
    elif sum(games[:2]) == 1: 
        win, loss = 1 + games[2], 1 + (1 - games[2]) # if the first two games are tied, the third game decides it
    else:
        win, loss = 2, 0 # otherwise deck 1 wins the first two games, so the result is 2-0
    
    return win, loss

def simulate_tournament(decks):

    '''input: a numpy array of the decks in the tournament (8x23)
       output: game_records, the records for each deck (numpy array, 8x2)'''

    records = np.zeros((8,2))
    indices = list(range(8))
    for i in range(3):
        np.random.shuffle(indices)
        shuffled_decks = decks[indices]
        for j in range(4):
            deck0_idx, deck1_idx = indices[2*j], indices[2*j+1]
            deck0, deck1 = shuffled_decks[2*j,:], shuffled_decks[2*j + 1, :]
            win, loss = simulate_match(deck0, deck1)
            records[[deck0_idx, deck1_idx],:] += [[win, loss], [loss, win]]
            
    return records

def simulate_spikein_dataset(trials):
    '''Simulates cube datasets with set tournaments numbers and an input number of trials.'''

    # create random strengths for each card, then for 20 random cards, overwrite their strength to 10 (spikein)
    cube_strengths = stats.norm.rvs(5,2, size = 450)
    spike_in = np.random.choice(range(450), size = 20, replace = False)
    cube_strengths[spike_in] = 10

    match_data = defaultdict(lambda: {'average_rank':0, 'percentage':0, 'average_games':0})
    # loop through number of tournaments
    for n in [100, 500, 1000, 10000]:
        print('On tournament #', n)
        spikeinrank_storage, percentage, average_games = [], [], []

        for trial in range(trials):
            if trial % 5 == 0: print('On trial {} of {}'.format(trial, trials))
            # initialize winrates
            winrates = np.array([np.array([float(0),float(0)]) for i in range(450)])

            # complete n tournaments
            for i in range(n):

                # choose 45 cards randomly for each deck, then for each deck choose the top 23 cards
                decks = np.random.choice(range(450), size = (8,45), replace = False)
                deck_strengths = cube_strengths[decks]
                col = np.argsort(-deck_strengths)[:,:23]
                row = np.tile(np.arange(0,8).reshape(8,1), (1,23))
                decks, deck_strengths = decks[row, col], deck_strengths[row, col]

                # simulate a tournament with the current deck_strengths
                records = simulate_tournament(deck_strengths)

                # update card records based on results of tournament
                for j in range(8):
                    cards = decks[j,:]
                    winrates[cards] += records[j,:]

            # extract the average number of games and calculate card win rates        
            games = np.mean(np.sum(winrates, axis = 1))
            wins, total = winrates[:,0], np.sum(winrates, axis = 1)
            card_winrates = np.divide(wins, total, out=np.zeros_like(wins), where=total!=0)

            # calculate the ranks of the spike in cards
            temp = np.argsort(-card_winrates)
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(card_winrates))
            spikein_ranks = ranks[spike_in]

            # update storage variables
            spikeinrank_storage.extend(spikein_ranks)
            percentage.append(sum(spikein_ranks < 100)/len(spikein_ranks))
            average_games.append(np.average(games))

        match_data[n]['average_rank'] = np.average(spikeinrank_storage)
        match_data[n]['percentage'] = percentage
        match_data[n]['average_games'] = np.average(average_games)

    # write results
    with open('simulation_results.txt', 'w') as sim_results:
        for i in [100, 500, 1000, 10000]:
            sim_results.write('Number of tournaments: {}\n'.format(i))
            sim_results.write('Average spikein rank: {}\n'.format(match_data[i]['average_rank']))
            sim_results.write('Average % of spikeins in top 100: {}\n'.format(np.mean(match_data[i]['percentage'])))
            sim_results.write('Average # of games played: {}\n'.format(match_data[i]['average_games']))
            sim_results.write('=============================\n')

    return 0

def simulate_rankings(trials):
    '''Simulates then plots estimated rankings vs. true rankings.'''

    cube_strengths = stats.norm.rvs(5,2, size = 450)
    
    # loop through number of tournaments
    rank_storage = []

    for trial in range(trials):
        if trial % 5 == 0: print('On trial {} of {}'.format(trial, trials))
        # initialize winrates
        winrates = np.array([np.array([float(0),float(0)]) for i in range(450)])

        # complete 10000 tournaments
        for i in range(10000):

            # choose 23 cards randomly for each deck
            decks = np.random.choice(range(450), size = (8,23), replace = False)
            deck_strengths = cube_strengths[decks]

            # simulate a tournament with the current deck_strengths
            records = simulate_tournament(deck_strengths)

            # update card records based on results of tournament
            for j in range(8):
                cards = decks[j,:]
                winrates[cards] += records[j,:]

        # extract the average number of games and calculate card win rates        
        wins, total = winrates[:,0], np.sum(winrates, axis = 1)
        card_winrates = np.divide(wins, total, out=np.zeros_like(wins), where=total!=0)

        # calculate the estimated ranks of the cards and appends
        temp = np.argsort(-card_winrates)
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(card_winrates))
        rank_storage.append(ranks)

    # determine the true rankings
    temp = np.argsort(-cube_strengths)
    true_ranks = np.empty_like(temp)
    true_ranks[temp] = np.arange(len(cube_strengths))
    
    # determine the average estimated rank for each card. Match each average estimated rank with the true rank.
    average_rank = np.array([np.average([rank_storage[i][j] for i in range(trials)]) for j in range(450)])
    rank_ordering = np.argsort(-true_ranks)
    sim_ranks = [x for _,x in sorted(zip(true_ranks,average_rank))]
    true_ranks = list(range(450))

    # determine confidence interval based on simulations
    upper, lower = [], []
    for i in rank_ordering[::-1]:
        distribution = [rank_storage[k][i] for k in range(trials)]
        bottom, top = np.percentile(distribution, [2.5, 97.5])
        upper.append(top)
        lower.append(bottom)

    # plot results
    ax, fig = plt.subplots(1,1, figsize = (10,6))
    plt.fill_between(range(450), lower, upper, alpha = 0.5, label = '95\% Confidence Interval')
    plt.plot(true_ranks, sim_ranks, label = 'Average Estimated Rank', color = 'black')
    plt.xlabel('True Rank', fontsize = 18)
    plt.ylabel('Estimated Rank', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.savefig('True_vs_Simulated_Rank_script.png', dpi = 300)

    return 0

def main():

    parser = argparse.ArgumentParser(description='simulates cube tournaments with spikeins and ranking determination.')
    parser.add_argument('-t','--trials', type=int, metavar='\b', help = 'number of trials to run', required = True)
    args = parser.parse_args()

    trials = args.trials

    # simulate spikein dataset
    simulate_spikein_dataset(trials)

    # simulate our ability to rank cards
    simulate_rankings(trials)

if __name__ == '__main__':
    main()