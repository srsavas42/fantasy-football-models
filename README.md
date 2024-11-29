# Fantasy Football Modelling

This project was undertaken with my interests for both fantasy football and statistical modelling in mind. To get data, I forked an initial repository. In revisiting the project, I found a python library that includes more data, allowing me to expand my analysis. 

The models can be found in the "models" folder.

## The Plan
Below is my general plan for creating a fantasy football analysis, including optimizing for draft strategy, player prediction, and weekly start/sit.

### 1: Draft Value Analysis
Fantasy football leagues start with a draft, where NFL players are selected to be on different teams. During a week, a team can start 1 QB, 2 RBs, 2 WRs, 1 TE, and 1 FLEX (RB/WR/TE). Each week, teams are placed head-to-head, where the team with the most points will win.

#### Tier-Based Relative Value
This plan is to first get the relative value for each position, with the objective of finding the biggest differences in player tier performance (i.e. top 4 RBs vs top 8). Tier gaps aim to compare the difference in value between positions at any point in the draft to determine which position should be prioritized. [DONE]

#### Pre-Season Expected Value
The next step is to take the expected value of each player. This entails mapping pre-season rankings to end-of-season rankings to create a distribution of outcomes and an expected value for their scoring. This is important because pre-season and post-season rankings are always significantly different, and understanding the risk/reward trade-off for a player is important to building a team.

#### Mid-Draft Trade-offs
The final step is to get the comparative value at a particular draft position relative to the next set of players I would be able to draft. For example, is it better to take a QB this round, or should I wait 3 more rounds for a QB in the next tier. This is how I can turn my analysis into actionable draft strategy.

### 2: Volume Prediction
One core tenent of fantasy football is that opportunity is king: players with more opportunities to touch the ball have more opportunities to score points. So, I want to do an analysis to try and predict the volume a player would get over the course of the season. I have identified the following critical factors:

1) Depth Chart Position: A player who is starting at the beginning of the season is more likely to get significant opportunity than a player who is a backup.
2) Coach Scheme: Coaches have different tendencies --- one team might like to throw the ball more, another wants to run. This impacts the opportunity players at different positions have to get the ball.
3) Injury Risk: Individual players get hurt, which limits their opportunity to play and score. However, when a starter gets hurt, their backup steps into the starting role, changing the dynamics of volume.

However, changes in volume can come with side effects, such as potentially decreasing efficiency. So, I would like to do some exploratory analysis on these relationships to get a better understanding of individual prediction.

### 3: Weekly Outcomes
As stated earlier, players have to choose a starting lineup. However, all players can change their starting lineup on different weeks. So, I would like to create a model to predict the distribution of outcomes for any given week such that I can choose a starting lineup by optimizing for various parameters such as expected point totals or upside.

# Fantasy Football Data Sets

If you are looking to run the scripts we've provided for locally updating data, clone this repo and install dependencies.

    pip install -r requirements.txt

## Strength of Schedule data
Strength of Schedule data is available in the sos directory. Data is available going back to 1999. To load this data in pandas using the following the following url format:
https://raw.githubusercontent.com/fantasydatapros/data/master/sos/{year}.csv

For example, in pandas do the following:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/sos/1999.csv', index_col=0)
    df.index = df.index.rename('Team')

## Weekly Fantasy Stats
Weekly stats going back to 1999 are available are exposed through the following url format

https://raw.githubusercontent.com/fantasydatapros/data/master/weekly/{year}/week{week}.csv

To grab weekly data for year 2019, week 1 in pandas, you would do:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/weekly/2019/week1.csv')

## Yearly Fantasy stats
Yearly fantasy stats are available going back to 1970.

The url format:
https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/{year}.csv

To grab yearly data for 2019 in pandas, do the following:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/2019.csv')




