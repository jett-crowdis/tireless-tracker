# Analyzing your own Cube drafts

This code accompanies the Tireless Tracker article Analyzing your own Cube drafts. There are two scripts in this directory:

1. `analyze_decklists.py`: this script analyzes an input folder of decklists and outputs various analyses.
2. `simulations.py`: this script contains the code that generates the simulations I used in the article. 

## `analyze_decklists.py`

### Inputs and Outputs:

The script takes in one required argument and three optional arguments:

* `deck_folder` (required): The relative path of the folder that contains your decklists.
* `filter` (optional, default 0): Cards must be present above this filter to be included in analysis. Useful for when some cards have minimal representation.
* `date` (optional, default False): Boolean for if the decklists have associated date information. If `True`, the script will perform a rolling average analysis of archetype winrates.
* `window` (optional, default = 100): The size of the sliding window for use if `date` argument is `True`.

The script outputs a variety of files:

* `Card_Analysis_{}%.csv`: these csv files contain the results of the individual card analysis. The same data is shown in three ways: sorting by win rate (Win%), normalized win rate (Norm%), and maindeck rate (Main%). Be careful when interpreting these files -- it is highly unlikely that you have enough decklists for them to be truly meaningful.
* `Color_Analysis.csv`: csv file that contains the results of color distribution analysis (in decklists and in cards).
* `Archetype_Analysis.csv`: csv file that contains the results of archetype win rates.
* `Archetype_Winrates.png`: if `date` in inputs is `True`, this file contains a graph depicting the win rates over time. If `date` is False, it is not created.
* `Misspellings.txt`: if the script encounters a card not found in Scryfall's database, it will skip it and write the card and file name to this file.
* `Cube_Analysis.xlsx`: an excel file that contains all the above information together.

### Usage
Place your folder of decklists in the same directory as `analyze_decklists.py`. Then run the following on the command line (requires Python to be installed)

__Basic Usage__
```
python analyze_decklists.py --deck_folder {your folder name}
```
__Optional Usage__

To specify a filter or to do timecourse analysis, use the following syntax. You can use a filter without using the date option
```
python analyze_decklists.py --deck_folder {your folder name} --filter {filter} --date {True or False} --window {window size} 
```
For example, you can regenerate the analysis used in the article by cloning this repo, navigating to the repo directory, and using 
```
python analyze_decklists.py --deck_folder Article_Example/Personal_Decklists_Updated --filter 20 --date True --window 70
```

### Decklist Formatting

The script expects decklists to be formatted in a specific way. This formatting is very sensitive and the script will break if a decklist is not formatted correctly. Check the decklists in `Personal_Decklists_Updated` for an example. Specifically, the format is:
```
Colors: WUBRG
Archetype: Superarchetype_Subarchetype
Record: MatchWins-MatchLosses
Games: GameWins-GameLosses

1 Goblin Guide
.
.
.

1 Inkwell Leviathan
.
.
.
```
The maindeck comes first, then the sideboard, separated by a line break. A few notes:

* The script does not recognize lowercase colors as splashed. Gb and GB are treated the same way.
* The super archetype belongs to every deck and the subarchetype is optional (ie Control_Reanimator). They must be separated by `_`. When analyzing win rates, the script will analyze a decklist containing a superarchetype and a subarchetype as counting towards both.
* Record and Games contain the same information on different levels. A deck that goes 3-0 can have a game record ranging from 6-0 to 6-3. Currently the script only analyzes game records - this will change soon.
* The sideboard is optional. The script will restrict calculating the maindeck % of a card to decklists with sideboards. 

### Note on Archetype Analysis:

The script will analyze "pure" archetypes as well as combined archetypes. This means that if a deck list is categorized as Control_Reanimator, it will count towards Control and Reanimator, but not "Pure Control". Only a decklist categorized as Control alone will count towards "Pure Control".

### Note on Dates: 

If you would like the script to do a time course analysis of your decklists, the "dates" are simply integers added to the end of files. See the example data used for the article for details. The dates must be added in this way.

## `simulations.py`

This script generates the cube simulation used in the article. There are two functions contained within this script, both of which are run when running the script. Fundamentally, the simulation works by simulating tournaments. Each card is assigned a random "strength" from about 0 - 10 (normally distributed with mean 5 and standard deviation 2). To simulate a tournament, 45 cards are randomly given to each "drafter". Then each drafter selects the 23 highest strength cards to make their deck. When two decks play against each other, they win at a rate proportional to their total deck strength. Each tournament simulates 3 different matches. 

### Usage

The script takes one required input

* `trials`: the number of trials that the simulation is run for.

The script outputs two files:

* `simulation_results.txt`: Contains the results of the the spike-in simulation. 
* `True_vs_Simulated_Rank_script.png`: Contains plot showing true vs. simulated rank for a 10,000 tournament simulation ran for the number of trials specified.

### `simulate_spikein_dataset`

This function takes in a number of input trials and simulates a card win rate dataset with simulated tournaments for that many trials, with spikeins. It will simulate a 100, 500, 1000, and 10,000 tournament dataset. For example, if 10,000 is specified for trials, it will conduct a 100, 500, 1000, and 10000 tournament dataset, and then repeat this process 10,000 times, averaging the spikein ranks over trials. It then writes the results to `simulation_results.txt`.

### `simulate_rankings`

This function simulates a 10,000 tournament dataset with as many trials as specified, averaging the results over trials. It then produces a graph that depicts the true ranking of cards vs. the simulated ranking. For the example used in the article, I used `trials = 20,000`, which takes about 3 hours to run in total. Note that for this function, the top 23 cards are not selected and all 45 cards are used. This is because selecting the top 23 cards eliminates the ability to evaluate very weak cards, which I wanted to demonstrate with the graph.
