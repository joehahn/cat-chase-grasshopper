# grid_walker

by Joe Hahn,<br />
jmh.datasciences@gmail.com,<br />
3 February 2018<br />
git branch=master


### Summary:
This grid_walker demo uses Q-learning to teach a neural net AI how to navigate an agent
about a very simple 6x6 grid, guiding it towards a goal while avoiding obstacles and hazards.

This version of grid_walker was adapted from a blog post by Outlace,
http://outlace.com/rlpart3.html. Outlace's original code is somewhat a mess,
while the version provided below is (I think) significantly less a mess.
Nonetheless Outlace's discussion of the Q-learning algorithm is excellent and worth a read.

### Setup:

Clone this repo:

    git clone https://github.com/joehahn/grid_walker.git
    cd grid_walker

Note that I am executing grid_walker on a Mac laptop where I've installed
Anaconda python 2.7 plus a few other needed libraries via:

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
    chmod +x ./Miniconda2-latest-MacOSX-x86_64.sh
    ./Miniconda2-latest-MacOSX-x86_64.sh -b -p ~/miniconda2
    ~/miniconda2/bin/conda install -y jupyter
    ~/miniconda2/bin/conda install -y keras

### Execute:

Start Jupyter notebook via

    jupyter notebook

and load the grid_walker.ipynb notebook > Run.

### Results:

Grid_walker is a simple game that consistes of 4 objects: an agent A, wall W, pit P, and goal G,
with these objects inhabiting a 6x6 grid. In the following, a 6x6 string array is used
to track and display the locations of all 4 objects:
![](figs/grid.png)
Only the agent is mobile while the other 3 objects P G and W remain at the same locations
during the every game.

Using the functionality provided in grid_walker.py, the agent can be instructed to move
to an adjacent grid cell, and in the following the agent A was
instructed to move one cell to the right, which also generated a small reward
ie cost of -1
![](figs/move.png)
The agent is then moved up and onto the goal G which generates a reward of 10
![](figs/goal.png)
This also changes the game_state from 'running' to 'goal' to indicate the game conclusion.
The pit P is a hazard, and moving the agent there ends the game with
reward -10, and the agent is not allowed to move onto wall W or beyond the 6x6 grid,
and trying do so generates a reward of -3. A game also ends if the agent moves
more than max_moves=36 times.

The purpose of the grid_walker demo is to build a neural-network based AI that will
advise the agent on how to navigate across this grid towards goal G without bumping to
the hazards W or P. This demo uses keras to build a two-hidden-layer net that are
each composed of 36 neurons
![](figs/net.png)
For additional details see the build_model() function,
https://github.com/joehahn/grid_walker/blob/master/grid_walker.py#L146#L161,
which is where keras + tensorflow are used to assemble that neural net model.

The model is then trained using an epsilon-greedy Q-learning algorithm:
![](figs/train.png)
Initially the agent wanders the grid randomly while gathering mostly negative rewards.
But in later training games the random movements are ramped down as the AI learns how to 
nudge the agent towards the direction of maximum future rewards; see the source
https://github.com/joehahn/grid_walker/blob/master/grid_walker.py#L164#L248
for more details on how Q-learning is used to train this AI.

To test the abilities of the
trained neural network model _trained\_model_, call the test_model() function
which plays a complete game of grid_walker with _trained\_model_ being used to select
the agent's move at every turn:
![](figs/test.png)
Success! Repeating this test will show that the model quite successful at
guiding the randomly-positioned agent A around hazards W and P and towards goal G.

However success is not 100%, and the following plays numerous grid_walker games
in order to chart those initial agent-positions that ultimately result in a win
for the AI (which is signified in the chart below via a _g_ symbol since the AI delivered
the agent into the Goal) and which
agent starting-positions result in a loss (ie the AI guided the agent into pit _p_ or else 
agent wandered the grid until the available moves _m_ where exhausted): 
![](figs/grid_test.png)
Note the _m_ cell, which appears to be 'shadowed' by wall W; agents starting here fail
to find goal G before exhausting all of there turns. Similarly, agents starting
in the two _p_ cells seem to be in the shadow of pit P, and the AI drives
these agents into the pit instead of the goal.
However the fix is simple, just double the number of games played when training
the model; set N_training_games=3000 and rerun to show that
the retrained AI now wins 100% of the time for all possible
initial positions:
![](figs/retrain.png)
:thumbsup:

