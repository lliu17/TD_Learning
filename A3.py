import random

class WindyGrid:
    def __init__(self, dimension, wind_arr, start, goal):
        self.LEARNING_RATE = 0.1
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
    
    def atGoal(self):
        return self.loc == self.goal

    def addWind(self):
        # print("self.loc[1]", self.loc[1])
        wind = self.wind_arr[self.loc[1]]
        print("wind =", wind)
        newLoc = (self.loc[0], self.loc[1] - wind)
        # if not in bound after adding wind, just "blow" it to as high as possible
        if not self.inBound(newLoc[0], newLoc[1]):
            newLoc = (self.loc[0], 0)
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
            # print("2, right")
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
    
    def randomAction(self):
        action = random.randrange(4)
        print("taking random action: ", self.numberToAction(action))
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
            print("random action")
            return self.randomAction()
        else:
            return self.greedyAction()

def sarsa_on_policy(grid, total_steps):
    time_step = 0
    epsilon = 0.1
    finished_episodes = 0
    finished_episodes_list = [0]

    while time_step <= total_steps:
        state1 = grid.loc
        action1 = grid.getEpsilonGreedyAction(epsilon)

        while not grid.atGoal() and time_step <= total_steps:
            reward = grid.takeAction(action1)
            state2 = grid.loc
            action2 = grid.getEpsilonGreedyAction(epsilon)

            grid.updateQTable(state1, action1, reward, state2, action2)
            state1 = state2
            action1 = action2
            time_step += 1
            finished_episodes_list.append(finished_episodes)
        
        if grid.atGoal():
            finished_episodes += 1
            grid.reset()

    return (time_step, finished_episodes)

wind_arr = [0, 1]
mini = WindyGrid((2, 2), wind_arr, (0, 0), (1, 1))
(time_step, finished_episodes) = sarsa_on_policy(mini, 5)
print("time_step", time_step)
print("finished_episodes", finished_episodes)
mini.print()

# sample grid from text book
# wind_arr = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# grid = WindyGrid((7, 10), wind_arr, (3,0), (3, 7))
# grid.print()

#         col0    col1
# row0    S 
# row1            G
# wind    0       1