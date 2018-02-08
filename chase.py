#!/usr/bin/env python

#hopper.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#4 February 2018
#
#this ...  
#to execute:    ./hopper.py

#imports
import numpy as np
import random
import copy
from collections import deque

#initialize the environment = dict containing all constants that describe the system
def initialize_environment(rn_seed, max_moves, max_distance):
    random.seed(rn_seed)
    actions = [0, 1, 2, 3, 4]
    acts = ['slow-left', 'slow-right', 'fast-left', 'fast-right', 'turnaround']
    environment = {'rn_seed':rn_seed, 'max_moves':max_moves, 'actions':actions, 'acts':acts,
        'max_distance':max_distance}
    return environment

#initialize state with bug at origin and cat randomly placed
def initialize_state(environment):
    bug = np.array([0.0, 0.0])
    stdev = 2.0
    x = random.normalvariate(0.0, stdev)
    y = random.normalvariate(0.0, stdev)
    cat = np.array([x, y])
    delta_bug = bug - cat
    state = {'cat':cat, 'bug':bug}
    bug_distance, bug_direction_angle = get_separation(state)
    cat_direction_angle = 0.0
    state['bug_distance'] = bug_distance
    state['bug_direction_angle'] = bug_direction_angle
    state['cat_run_direction_angle'] = bug_direction_angle
    return state

#calculate the bug-cat separation and direction
def get_separation(state):
    cat = state['cat']
    bug = state['bug']
    delta_bug = bug - cat
    bug_direction_angle = np.arctan2(delta_bug[1], delta_bug[0])
    bug_distance = np.sqrt((delta_bug**2).sum())
    return bug_distance, bug_direction_angle

#move the bug
def move_bug(state):
    #random component of bug's movement
    stdev = 2.0
    random_distance = random.normalvariate(0.0, stdev)
    pi = np.pi
    random_angle = random.uniform(-pi, pi)
    bug_dx = random_distance*np.cos(random_angle)
    bug_dy = random_distance*np.sin(random_angle)
    #systematic component is away from cat
    separation, angle = get_separation(state)
    softening_distance = 0.4
    leap_distance = 1.0/np.sqrt(softening_distance**2 + separation**2)
    leap_angle = angle + np.pi
    bug_dx += leap_distance*np.cos(leap_angle)
    bug_dy += leap_distance*np.sin(leap_angle)
    bug_displacement = np.array([bug_dx, bug_dy])
    return bug_displacement

#move cat
def move_cat(state, action):
    #cat moves slow or fast
    slow_speed = 0.1
    fast_speed = 0.5
    if ((action == 0) or (action == 1)):
        speed = slow_speed
    if ((action == 2) or (action == 3)):
        speed = fast_speed
    #cat turns left or right about 25 degrees
    delta_angle = 2*np.pi/15
    if ((action == 0) or (action == 2)):
        #slight left turn
        cat_run_direction_angle = state['cat_run_direction_angle'] + delta_angle
    if ((action == 1) or (action == 3)):
        #slight right turn
        cat_run_direction_angle = state['cat_run_direction_angle'] - delta_angle
    if (action == 4):
        #cat turns around
        cat_run_direction_angle = state['cat_run_direction_angle'] + np.pi
        if (cat_run_direction_angle > np.pi):
            cat_run_direction_angle -= 2.0*np.pi
        speed = slow_speed
    timestep = 1.0
    distance = speed*timestep
    dx = distance*np.cos(cat_run_direction_angle)
    dy = distance*np.sin(cat_run_direction_angle)
    cat_displacement = np.array([dx, dy])
    return cat_displacement, cat_run_direction_angle

#move bug and cat
def update_state(state, action):
    #bug's move is probabilistic
    distance, angle = get_separation(state)
    softening_distance = 0.05
    bug_move_probability = 1.0/np.sqrt(1.0 + (distance/softening_distance)**2)
    bug_displacement = move_bug(state)
    if (random.uniform(0.0, 1.0) < bug_move_probability):
        pass
    else:
        bug_displacement *= 0
    #cat's move
    cat_displacement, cat_run_direction_angle = move_cat(state, action)
    #update state
    next_state = copy.deepcopy(state)
    next_state['bug'] += bug_displacement
    next_state['cat'] += cat_displacement
    next_state['cat_run_direction_angle'] = cat_run_direction_angle
    bug_distance, bug_direction_angle = get_separation(next_state)
    next_state['bug_distance'] = bug_distance
    next_state['bug_direction_angle'] = bug_direction_angle
    return next_state

#calculate reward
def get_reward(state, environment):
    bug_distance, bug_direction_angle = get_separation(state)
    softening_distance = 0.2
    reward = 1.0/np.sqrt(softening_distance**2 + bug_distance**2) - bug_distance/5
    max_distance = environment['max_distance']
    if (bug_distance > max_distance):
        reward -= -7.0
    return reward

#check game state = running, or too many moves
def get_game_state(N_moves, state, environment):
    game_state = 'running'
    max_moves = environment['max_moves']
    if (N_moves > max_moves):
        game_state = 'max_moves'
    bug_distance, bug_direction_angle = get_separation(state)
    max_distance = environment['max_distance']
    if (bug_distance > max_distance):
        game_state = 'max_distance'
    return game_state

#play one game, with game history stored in memories queue
def play_game(environment, strategy, model=None):
    max_moves = environment['max_moves']
    memories = deque(maxlen=max_moves+1)
    state = initialize_state(environment)
    N_moves = 1
    game_state = get_game_state(N_moves, state, environment)
    while (game_state == 'running'):
        cat_run_direction_angle = state['cat_run_direction_angle']
        bug_direction_angle = state['bug_direction_angle']
        bug_cat_relative_angle = bug_direction_angle - cat_run_direction_angle
        if (bug_cat_relative_angle > np.pi):
            bug_cat_relative_angle -= 2*np.pi
        if (bug_cat_relative_angle < -np.pi):
            bug_cat_relative_angle += 2*np.pi
        if (strategy == 'slow'):
            if (bug_cat_relative_angle > 0):
                #advance slowly with slight turn to left
                action = 0
            else:
                #advance slowly with slight turn to right
                action = 1
        if (strategy == 'fast'):
            if (bug_cat_relative_angle > 0):
                #advance rapidly with slight turn to left
                action = 2
            else:
                #advance rapidly with slight turn to right
                action = 3
        if (strategy == 'smart'):
            state_vector = state2vector(state)
            Q = model.predict(state_vector, batch_size=1)
            action = np.argmax(Q)
        state_next = update_state(state, action)
        reward = get_reward(state_next, environment)
        game_state = get_game_state(N_moves, state_next, environment)
        memory = (state, action, reward, state_next, game_state)
        memories.append(memory)
        state = copy.deepcopy(state_next)
        N_moves += 1
    return memories

#convert memories queue into separate timeseries arrays
def memories2arrays(memories):
    cat_list = []
    bug_list = []
    actions_list = []
    rewards_list = []
    bug_distances_list = []
    bug_direction_angles_list = []
    cat_run_direction_angles_list = []
    for memory in memories:
        state, action, reward, state_next, game_state = memory
        cat_list += [state_next['cat']]
        bug_list += [state_next['bug']]
        actions_list += [action]
        rewards_list += [reward]
        bug_distances_list += [state_next['bug_distance']]
        bug_direction_angles_list += [state_next['bug_direction_angle']]
        cat_run_direction_angles_list += [state_next['cat_run_direction_angle']]
    cat = np.array(cat_list)
    bug = np.array(bug_list)
    actions = np.array(actions_list)
    rewards = np.array(rewards_list)
    bug_distances = np.array(bug_distances_list)
    bug_direction_angles = np.array(bug_direction_angles_list)
    cat_run_direction_angles = np.array(cat_run_direction_angles_list)
    turns = np.arange(len(rewards))
    return cat, bug, actions, rewards, bug_distances, bug_direction_angles, cat_run_direction_angles, turns

#build neural network
def build_model(N_inputs, N_neurons, N_outputs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(Dense(N_neurons, input_shape=(N_inputs,)))
    model.add(Activation('relu'))
    model.add(Dense(N_neurons))
    model.add(Activation('relu'))
    model.add(Dense(N_outputs))
    model.add(Activation('linear'))
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    return model

#convert state into a numpy array of agents' x,y coordinates
def state2vector(state):
    cat_run_direction_angle = state['cat_run_direction_angle']
    bug_direction_angle = state['bug_direction_angle']
    bug_distance = state['bug_distance']
    v = np.array([cat_run_direction_angle, bug_direction_angle, bug_distance])
    return v.reshape(1, len(v))

#train model
def train(environment, model, N_training_games, max_distance, gamma, memories, batch_size, debug=False):
    epsilon = 1.0
    for N_games in range(N_training_games):
        rn_seed = N_games
        max_moves = int(1000*((N_games*1.0/N_training_games)**2) + 10)
        environment = initialize_environment(rn_seed, max_moves, max_distance)
        state = initialize_state(environment)
        state_vector = state2vector(state)
        N_inputs = state_vector.shape[1]
        experience_replay = True
        N_moves = 1
        if (N_games > N_training_games/10):
            #agent executes random actions for first 10% games, after which epsilon ramps down to 0.1
            if (epsilon > 0.1):
                epsilon -= 1.0/(N_training_games/2)
        game_state = get_game_state(N_moves, state, environment)
        while (game_state == 'running'):
            state_vector = state2vector(state)
            #predict this turn's possible rewards Q
            Q = model.predict(state_vector, batch_size=1)
            #choose best action
            if (np.random.random() < epsilon):
                #choose random action
                action = np.random.choice(environment['actions'])
            else:
                #choose best action
                action = np.argmax(Q)
            #get next state
            state_next = update_state(state, action)
            state_vector_next = state2vector(state_next)
            #predict next turn's possible rewards
            Q_next = model.predict(state_vector_next, batch_size=1)
            max_Q_next = np.max(Q_next)
            reward = get_reward(state_next, environment)
            game_state = get_game_state(N_moves, state_next, environment)
            #add next turn's discounted reward to this turn's predicted reward
            Q[0, action] = reward
            if (game_state == 'running'):
                Q[0, action] += gamma*max_Q_next
            else:
                if (debug):
                    print '======================='
                    print 'game number = ', N_games
                    print 'move number = ', N_moves
                    print 'final action = ', environment['acts'][action]
                    print 'final reward = ', reward
                    print 'epsilon = ', epsilon
                    print 'game_state = ', game_state
                else:
                    print '.',
            if (experience_replay):
                #train model on randomly selected past experiences
                memories.append((state, action, reward, state_next, game_state))
                memories_sub = random.sample(memories, batch_size)
                statez = [m[0] for m in memories_sub]
                actionz = [m[1] for m in memories_sub]
                rewardz = [m[2] for m in memories_sub]
                statez_next = [m[3] for m in memories_sub]
                game_statez = [m[4] for m in memories_sub]
                state_vectorz_list = [state2vector(s) for s in statez]
                state_vectorz = np.array(state_vectorz_list).reshape(batch_size, N_inputs)
                Qz = model.predict(state_vectorz, batch_size=batch_size)
                state_vectorz_next_list = [state2vector(s) for s in statez_next]
                state_vectorz_next = np.array(state_vectorz_next_list).reshape(batch_size, N_inputs)
                Qz_next = model.predict(state_vectorz_next, batch_size=batch_size)
                for idx in range(batch_size):
                    reward = rewardz[idx]
                    max_Q_next = np.max(Qz_next[idx])
                    action = actionz[idx]
                    Qz[idx, action] = reward
                    if (game_statez[idx] == 'running'):
                        Qz[idx, action] += gamma*max_Q_next
                model.fit(state_vectorz, Qz, batch_size=batch_size, epochs=1, verbose=0)
            else:
                #teach model about current action & reward
                model.fit(state_vector, Q, batch_size=1, epochs=1, verbose=0)
            state = state_next
            N_moves += 1
    return model
