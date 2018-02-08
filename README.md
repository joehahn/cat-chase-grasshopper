# cat-chase-grasshopper

by Joe Hahn,<br />
jmh.datasciences@gmail.com,<br />
8 February 2018<br />
git branch=master


### Summary:
This cat-chase-grasshopper demo uses Q-learning to teach a neural net AI how to instruct
a virtual cat to chase after a virtual bug that hops away as the cat approaches.

A secondary goal of this demo is to see if Q-learning can be used to solve
this optimization problem: imagine a box with two dials that can be used
to move an agent (the cat) towards the a moving target (the grasshopper). The goal
is to write code that turns the dials so that the agent-target separation stays
minimized, without knowing in advance how the agent responds to twists of either
dial. Solution: use Q-learning to teach a neural network how to turn the dials so that
the cat chases the grasshopper as closely as possible.

### Setup:

Clone this repo:

    git clone https://github.com/joehahn/cat-chase-grasshopper.git
    cd cat-chase-grasshopper

I am executing cat-chase-grasshopper on a Mac laptop where I've installed
Anaconda python 2.7 plus additional libraries via:

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
    chmod +x ./Miniconda2-latest-MacOSX-x86_64.sh
    ./Miniconda2-latest-MacOSX-x86_64.sh -b -p ~/miniconda2
    ~/miniconda2/bin/conda install -y jupyter
    ~/miniconda2/bin/conda install -y keras
    ~/miniconda2/bin/conda install -y seaborn

### Execute:

Start Jupyter notebook via

    jupyter notebook

and load the cat-chase-grasshopper.ipynb notebook > Run.

### Results:

Cat-chase-grasshopper is a 2D game composed of a virtual cat that is always chasing
a virtual grasshopper that hops away whenever the cat nears. The game is turn-based, and each
turn the cat can execute one of these actions: move slow or fast with a slight
25 degree turn to the left or right, or turn 180 degrees. So the cat has five possible
actions: move slow & turn left, move slow & turn right, move fast turn left, move fast
turn right, or turn around. Nearly straight motion is achieved by alternating
left and right turns.

The reason that the cat's movements are quantized is so that we can use Q-learning
to drive the cat towards the grasshopper. Q-learning is a fairly straightforward
and not-too-difficult reinforcement-learning algorithm, but one that is restricted
to discreet actions. If the cat was instead allowed a continuous choice of movements such as
variable speed or direction, then we would have to use much more complex algorithms
like actor-critic or DDPG, and that is not attempted here.
 
Helper functions are stored in chase.py, these functions are used to define
the game and initialize the bug & cat's x,y coordinates. Every turn the cat is given a
reward that increases as the bug-cat separation is reduced:
![](figs/reward_vs_separation.png)
This varies roughly as reward ~ 1/separation - separation/5, and these rewards will be used
to train the AI to to steer the cat towards the bug.

The bug's hopping motion is random, with the probability of the bug executing a hop
varying roughly as probability ~ constant/separation. A hop has two components, a
random hop of distance ~2 in any directon + a systematic component whose distance
is ~1/distance in the direction away from the cat.


The point of this demo is to use
Q-learning to train the cat to stay as near as possib
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

