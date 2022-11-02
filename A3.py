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
        self.greedyAction = {}      # map state --> greedy action

    def print(self):
        print("Windy Grid")
        print("Start:", self.start)
        print("Goal:", self.goal)
        print("Current Location:", self.loc)
        print()

    def inBound(self, row, col):
        if row >= self.dimension[0] or col >= self.dimension[1]:
            return False
        return True
    
    def reset(self):
        self.loc = self.start
    
    def atGoal(self):
        return self.loc == self.goal

    def addWind(self):
        wind = self.wind_arr[self.loc[1]]
        newLoc = [self.loc[0], self.loc[1] + wind]
        # if not in bound after adding wind, just "blow" it to as high as possible
        if not self.inBound(newLoc):
            newLoc = [self.loc[0], self.dimension[1] - 1]

    def determineAction(self):


    def takeAction(self, actionIdx):
        newLoc = None
        prev_state = self.loc
        if actionIdx == 0:
            print("0, up")
            newLoc = [self.loc[0], self.loc[1] + 1] 
        elif actionIdx == 1:
            print("1, down")
            newLoc = [self.loc[0], self.loc[1] - 1] 
        elif actionIdx == 2:
            print("2, right")
            newLoc = [self.loc[0] + 1, self.loc[1]] 
        else:
            print("3, left")
            newLoc = [self.loc[0] - 1, self.loc[1]] 
        # if not in bound, stay in place
        if self.inBound(newLoc[0], newLoc[1]):
            self.loc = newLoc
            self.addWind()
        
        reward = -1
        if self.atGoal():
            reward = 0

        self.updateQTable(prev_state, reward)

    def updateQTable(self, prev_state, action, curr_state, reward):
        
        if prev_state not in self.qTable:
            self.qTable[prev_state] = [0, 0, 0, 0]
        
        curr_state_value = 0
        if curr_state in self.qTable:
        
        arr = self.qTable[prev_state]
        arr[action] += self.LEARNING_RATE * (self.DEFAULT_REWARD + )
    
    def randomAction(self):
        currAction = random.randrange(4)
        return 
        self.takeAction(currAction)

    def greedyAction(self, state):
        if state not in self.qTable:
            self.randomAction()
        else:
            action_arr = self.qTable[state]
            action_value = max(action_arr)
            action = action_arr.index(action_value)
            # if "best action" so far is uninitialized:
            # find all uninitialized action and take a random one
            if action_value == 0:
                choices = []
                for i in range(len(action_arr)):    # constant iteration of 4
                    if action_arr[i] == 0:
                        choices.append(i)
                action = random.choice(choices)
            self.takeAction(action)

def sarsa_on_policy(grid, total_steps):
    time_step = 0
    default_reward = -1
    epsilon = 0.1
    finished_episodes = 0
    rewards = 0

    while time_step <= total_steps:
        while not grid.atGoal() and time_step <= total_steps:
            rand_num = random.random()
            if rand_num < epsilon:
                action = grid.randomAction()
            else:
                action = grid.greedyAction()
            grid.takeAction(action)
            time_step += 1
        
        if grid.atGoal():
            finished_episodes += 1
        grid.reset()

# sample grid from text book
# wind_arr = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# grid = WindyGrid((7, 10), wind_arr, (3,0), (3, 7))
# grid.print()
