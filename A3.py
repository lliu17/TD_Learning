import random
import numpy as np
import matplotlib.pyplot as plt

class WindyGrid:
    def __init__(self, dimension, wind_arr, start, goal):
        self.LEARNING_RATE = 0.5
        self.DEFAULT_REWARD = -1
        self.dimension = dimension
        self.wind_arr = wind_arr
        self.start = start
        self.goal = goal
        self.loc = start
        self.qTable = {}            # map (state, action) --> value
        # self.greedyAction = {}      # map state --> greedy action

    def print(self):
        print("Windy Grid")
        print("Start:", self.start)
        print("Goal:", self.goal)
        print("Current Location:", self.loc)
        print("qTable:", self.qTable)
        print()

    def numberToAction(self, actionIdx):
        if actionIdx == 0:
            return "0, up"
        elif actionIdx == 1:
            return "1, down"
        elif actionIdx == 2:
            return "2, right"
        else:
            return "3, left"
        
    def inBound(self, row, col):
        if row < 0 or row >= self.dimension[0] or \
           col < 0 or col >= self.dimension[1]:
            return False
        return True
    
    def reset(self):
        self.loc = self.start
        self.qTable = {}
    
    def resetLoc(self):
        self.loc = self.start

    def atGoal(self):
        if self.loc == self.goal:
            # print("=== AT GOAL ===")
            return True
        return False

    def addWind(self):
        # print("self.loc[1]", self.loc[1])
        wind = self.wind_arr[self.loc[1]]
        # print("wind =", wind)
        newLoc = (self.loc[0] - wind, self.loc[1])
        # if not in bound after adding wind, just "blow" it to as high as possible
        if not self.inBound(newLoc[0], newLoc[1]):
            newLoc = (0, self.loc[1])
        self.loc = newLoc

    def takeAction(self, actionIdx):
        newLoc = None
        if actionIdx == 0:
            # print("0, up")
            newLoc = (self.loc[0] - 1, self.loc[1])
        elif actionIdx == 1:
            # print("1, down")
            newLoc = (self.loc[0] + 1, self.loc[1])
        elif actionIdx == 2:
            # print("takeAction: 2, right")
            newLoc = (self.loc[0], self.loc[1] + 1) 
        else:
            # print("3, left")
            newLoc = (self.loc[0], self.loc[1] - 1) 
        # if not in bound, stay in place
        if self.inBound(newLoc[0], newLoc[1]):
            self.loc = newLoc
        self.addWind()
        
        reward = -1
        if self.atGoal():
            reward = 0
        return reward

    def updateQTable(self, state1, action1, reward, state2, action2):
        if state1 not in self.qTable:
            self.qTable[state1] = [0, 0, 0, 0]
        value_arr = self.qTable[state1]

        state2_action_value = 0
        if state2 in self.qTable:
            state2_action_value = self.qTable[state2][action2]
        
        value_arr[action1] += self.LEARNING_RATE * \
            (reward + state2_action_value - value_arr[action1])
    
    def updateQTable_n_step(self, state, action, G):
        if state not in self.qTable:
            self.qTable[state] = [0, 0, 0, 0]
        value_arr = self.qTable[state]
        value_arr[action] += self.LEARNING_RATE * (G - value_arr[action])

    def randomAction(self):
        action = random.randrange(4)
        # print("random action: taking random action: ", self.numberToAction(action))
        return action

    def greedyAction(self):
        state = self.loc
        if state not in self.qTable:
            return self.randomAction()
        else:
            action_arr = self.qTable[state]
            action_value = max(action_arr)
            action = action_arr.index(action_value)
            # print("action_value: ", action_value)
            # print("action: ", action)

            # if "best action" so far is uninitialized:
            # find all uninitialized action and take a random one
            if action_value == 0:
                choices = []
                for i in range(len(action_arr)):    # constant iteration of 4
                    if action_arr[i] == 0:
                        choices.append(i)
                action = random.choice(choices)
            return action

    def getEpsilonGreedyAction(self, epsilon):
        rand_num = random.random()
        if rand_num < epsilon:
            # print("Epsilon greedy: random action")
            return self.randomAction()
        else:
            # print("Epsilon greedy: greedy action")
            return self.greedyAction()

def sarsa_on_policy(grid, total_steps, n):
    time_step = 0
    epsilon = 0.1
    finished_episodes = 0
    finished_episodes_list = []

    while time_step < total_steps:
        state1 = grid.loc
        action1 = grid.getEpsilonGreedyAction(epsilon)

        # print("state1:", state1)
        # print("action1:", grid.numberToAction(action1))
        reward = grid.takeAction(action1)
        state2 = grid.loc
        action2 = grid.getEpsilonGreedyAction(epsilon)
        # print("reward:", reward)
        # print("state2:", state2)
        
        # print("action2:", grid.numberToAction(action2))
        # print()

        grid.updateQTable(state1, action1, reward, state2, action2)
        state1 = state2
        action1 = action2
        
        if grid.atGoal():
            finished_episodes += 1
            grid.resetLoc()
            state1 = grid.loc
            action1 = grid.getEpsilonGreedyAction(epsilon)
        
        time_step += 1
        finished_episodes_list.append(finished_episodes)

    return (time_step, finished_episodes, finished_episodes_list)

def Q_Learning(grid, total_steps, n):
    time_step = 0
    epsilon = 0.1
    finished_episodes = 0
    finished_episodes_list = []

    while time_step < total_steps:
        state1 = grid.loc
        action1 = grid.getEpsilonGreedyAction(epsilon)
        reward = grid.takeAction(action1)
        state2 = grid.loc
        action2 = grid.greedyAction()
        grid.updateQTable(state1, action1, reward, state2, action2)
        
        if grid.atGoal():
                finished_episodes += 1
                grid.resetLoc()
        
        time_step += 1
        finished_episodes_list.append(finished_episodes)
        
    return (time_step, finished_episodes, finished_episodes_list)

def n_step_sarsa(grid, total_steps, n):
    time_step = 0
    tao = 0 # state whose value will be updated
    epsilon = 0.1
    finished_episodes = 0
    finished_episodes_list = []
    num_updates = 0
    total_count = 0

    states = [grid.loc] # initialize state
    actions = [grid.getEpsilonGreedyAction(epsilon)]# initialize action
    rewards = [0]
    # total_steps -= time_step
    T = total_steps

    while tao < T - 1 and total_count < total_steps:
        # print("total count =", total_count)
        # print("=== time step = ", time_step)
        if time_step < T:
            # print("taking action and store rew, state")
            reward = grid.takeAction(actions[time_step])
            rewards.append(reward)
            states.append(grid.loc)
        
            if grid.atGoal():
            # if states[time_step + 1] == grid.goal:
                # print("--- AT GOAL")
                T = time_step + 1
                finished_episodes += 1
                grid.resetLoc() # reset episode after reaching goal
                states = [grid.loc]                              # initialize state
                actions = [grid.getEpsilonGreedyAction(epsilon)] # initialize action
                rewards = [0]
                # total_steps -= time_step
                # T = total_steps
                tao = 0
                # if tao >= T - 1:
                #     break
                # print("T = ", T, ", tao =", tao)
                time_step = -1
                # print("stuck here?")
            else:
                actions.append(grid.getEpsilonGreedyAction(epsilon))

        tao = time_step - n + 1
        # print("tao =", tao)
        if tao >= 0:
            G = 0
            i = tao + 1
            bound = min(tao + n, T)
            # print("i = ", i, "bound =", bound)
            # print("rewards =", rewards)
            while i <= bound:
                # G += discount ** (i - tao - 1) * rewards[i]
                G += rewards[i]
                i += 1
            
            if tao + n < T:
                # print("tao = ", tao)
                state = states[tao + n]
                # print("state = ", state)
                if state in grid.qTable:
                    action = actions[tao + n]
                    # G += discount ** n * grid.qTable[state][action]
                    G += grid.qTable[state][action]
            grid.updateQTable_n_step(states[tao], actions[tao], G)
            num_updates += 1
        # print("time_step", time_step)
        time_step += 1
        total_count += 1
        finished_episodes_list.append(finished_episodes)
    # print("total count:", total_count)
    print("num_updates:", num_updates)
    return (time_step, finished_episodes, finished_episodes_list)

def printOptimalPath(grid, i):
    print("Printing optimal path:")
    grid.resetLoc()
    length = 0
    while not grid.atGoal():
        state1 = grid.loc
        print("curr state:", state1, "\\\\")
        # action_arr = grid.qTable[state1]
        # print("action_arr:", [ "{:0.2f}".format(x) for x in action_arr])
        action1 = grid.greedyAction()
        print("Taking action:", grid.numberToAction(action1), "\\\\")
        reward = grid.takeAction(action1)
        # print("reward =", reward)
        state2 = grid.loc
        action2 = grid.greedyAction()
        grid.updateQTable(state1, action1, reward, state2, action2)

        # action = grid.getEpsilonGreedyAction(0.1)
        # # print("Taking action:", grid.numberToAction(action), "\\\\")
        # grid.takeAction(action)
        length += 1
        if length > 1000:
            raise ValueError("optimal path length exceeding 1000, \
                at experiment ", i) 
    print("length of optimal path:", length)

def optimalPath(grid, i):
    grid.resetLoc()
    length = 0
    while not grid.atGoal():
        state1 = grid.loc
        action1 = grid.greedyAction()
        reward = grid.takeAction(action1)
        state2 = grid.loc
        action2 = grid.greedyAction()
        grid.updateQTable(state1, action1, reward, state2, action2)
        length += 1

        if length > 1000:
            raise ValueError("optimal path length exceeding 1000, \
                at experiment ", i) 
    return length

def calcNewAvg(prev_avg, new_res, n):
    return ((prev_avg * n) + new_res) / (n + 1)

def calcAvg(master_list, new_list, experiment_id):
    for i in range(len(master_list)):
        master_list[i] = calcNewAvg(master_list[i], new_list[i], experiment_id)

def runExperiment(method, grid, num_experiments, total_steps, n):
    print("running:", method)
    optimal_path_length = 0
    finished_episodes_avg = 0
    finished_episodes_list_master = []
    for i in range(num_experiments):
        grid.reset()
        random.seed(i)
        (time_step, finished_episodes, finished_episodes_list) = method(grid, total_steps, n)
        if not finished_episodes_list_master:
            finished_episodes_list_master = finished_episodes_list
        else:
            calcAvg(finished_episodes_list_master, finished_episodes_list, i)
        
        # length = optimalPath(grid, i)
        finished_episodes_avg = calcNewAvg(finished_episodes_avg , finished_episodes, i)
        # optimal_path_length = calcNewAvg(optimal_path_length, length, i)

    # optimalPath(grid)
    # grid.print()
    # print("time_step", time_step)
    # print("finished_episodes", finished_episodes)

    print("number of experiments:", num_experiments)
    print("number of time steps for each experiment:", total_steps)
    print("finished_episodes_avg", finished_episodes_avg)
    print("optimal_path_length avg:", optimal_path_length)
    # print(finished_episodes_list_master)
    print()

    return finished_episodes_list_master


#         col0    col1
# row0    S 
# row1            G
# wind    0       1
# wind_arr = [0, 1]
# mini = WindyGrid((2, 2), wind_arr, (0, 0), (0, 1))
# (time_step, finished_episodes, finished_episodes_list) = sarsa_on_policy(mini, total_steps)
# mini.print()

# # sample grid from text book
wind_arr = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
grid = WindyGrid((7, 10), wind_arr, (3,0), (3, 7))

num_experiments = 100
total_steps = 8000
# finished_episodes_list_master_s = runExperiment(sarsa_on_policy, grid, num_experiments, total_steps)
# finished_episodes_list_master_q = runExperiment(Q_Learning, grid, num_experiments, total_steps)

# x_axis = np.arange(total_steps)
# plt.title("Finished Episodes over Time Steps")
# plt.plot(x_axis, finished_episodes_list_master_q, label="Q Learning")
# plt.plot(x_axis, finished_episodes_list_master_s, label="Sarsa")
# plt.xlabel('Time Steps')
# plt.ylabel('Finished Episodes')
# plt.legend()
# plt.show()

# random.seed(0)
# (time_step, finished_episodes, finished_episodes_list) = n_step_sarsa(grid, 6, 8000, 1)
# print("finished_episodes:", finished_episodes)
finished_episodes_list_master_n = runExperiment(n_step_sarsa, grid, num_experiments, total_steps, 1)
# grid.print()

x_axis = np.arange(len(finished_episodes_list_master_n))
plt.title("Finished Episodes over Time Steps")
plt.plot(x_axis, finished_episodes_list_master_n, label="3-step sarsa")
# plt.plot(x_axis, finished_episodes_list_master_q, label="Q Learning")
# plt.plot(x_axis, finished_episodes_list_master_s, label="Sarsa")
plt.xlabel('Time Steps')
plt.ylabel('Finished Episodes')
plt.legend()
plt.show()