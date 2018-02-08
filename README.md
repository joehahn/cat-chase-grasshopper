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

The bug's hopping motion is random, with the probability that the bug hops during a turn
varying as ~ constant/separation. The bug's hop has two components, a
random hop of distance of order ~2 in any directon + a systematic hop whose distance
is ~1/separation in the direction away from the cat.

But before we train the AI to steer the cat, lets play one game of cat-chase-grasshopper
using the 'slow' strategy where the cat is preprogrammed to advance towards the bug
at slow speeds, and the following plot shows the cat's trajectory (green dots)
as it chases the hopping grasshopper as it random-walks (blue dots).
![](figs/xy_slow.png)

The large translucent dots show the cat and bug's final positions, and initial
positions are the intermediate dense dots, and the plot below shows that rewards
that the cat accumulated during this game:
 ![](figs/rewards.png)

Now play this game again using the 'fast' strategy, here the cat is preprogrammed
to always turn towards the bug and advance at the faster speed that is 5x faster than slow:
![](figs/xy_fast.png)



:thumbsup:

