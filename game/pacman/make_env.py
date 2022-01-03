import sys
import gym
import numpy as np
from gym import spaces

import game.pacman.layout as layout
from game.pacman.pacman import readCommand
from game.pacman.pacman import ClassicGameRules
import game.pacman.textDisplay as textDisplay
from game.pacman.ghostAgents import RandomGhost as Ghost


class Agent:
    def __init__(self):
        self.name = ''

    def get(self):
        raise NotImplemented()


class Wrap_pacman():
    def __init__(self, args):
        #  , layout, pacman, ghosts, display, numGames, record, numTraining = 0, catchExceptions = False, timeout = 30
        self.args = args
        self.layout = layout.getLayout(args['game_name'])
        self.rules = ClassicGameRules(self.args['timeout'])
        self.pacman = Agent()
        self.ghosts = [Agent() for i in range(self.layout.getNumGhosts())]#[Ghost(i+1) for i in range(self.layout.getNumGhosts())]  # [Agent() for i in range(self.layout.getNumGhosts())]
        if self.args['quietGraphics']:
            display = textDisplay.NullGraphics()
        elif self.args['textGraphics']:
            textDisplay.SLEEP_TIME = self.args['frameTime']
            display = textDisplay.PacmanGraphics()
        else:
            import game.pacman.graphicsDisplay as graphicsDisplay
            display = graphicsDisplay.PacmanGraphics(self.args['zoom'], frameTime=self.args['frameTime'])
        self.beQuiet = False
        self.textDisplay = textDisplay.NotGraphics()
        self.videoDisplay = display
        self.rules.quiet = True
        self.catchExceptions = self.args['catchExceptions']
        self.done = True

        self.action2str = ['North', 'South', 'East', 'West', 'Stop']
        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts, display, self.beQuiet,
                                       self.catchExceptions)

        # gym-like info
        self.n = len(self.game.agents)
        self.action_space = [spaces.Discrete(len(self.action2str)) for i in range(self.n)]
        self.observation_space = [spaces.Box(low=0, high=1, shape=((self.layout.width + self.layout.height) * self.n + 18,), dtype=np.float32) if i == 0 else
                                  spaces.Box(low=0, high=1, shape=((self.layout.width + self.layout.height) * 2,), dtype=np.float32)
                                  for i in range(self.n)]

    def step(self, actions, done=None):
        assert not self.done, 'done!  step after reset'
        actions = [np.argmax(a) for a in actions]
        actions = [self.action2str[action] for action in actions]
        # ghost_action = []
        # ghost_action.append(actions[0])
        # for ghost in self.ghosts:
        #     action = ghost.getAction(self.game.state)
        #     ghost_action.append(action)
        # print(ghost_action)
        state, reward, done, info = self.game.step(actions)
        self.done = done
        done = [done for i in range(self.n)]
        return state, reward, done, info

    def reset(self, render=False):
        del self.game
        del self.rules
        del self.pacman
        del self.ghosts

        self.pacman = Agent()
        self.ghosts = [Agent() for i in range(self.layout.getNumGhosts())]#[Ghost(i+1) for i in range(self.layout.getNumGhosts())]

        self.rules = ClassicGameRules(self.args['timeout'])
        self.rules.quiet = True

        if render:
            display = self.videoDisplay
            self.rules.quiet = False
        else:
            display = self.textDisplay
            self.rules.quiet = True

        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts, display, self.beQuiet,
                                       self.catchExceptions)

        self.done = False

        return self.game.reset(render=render)

    def render(self):
        pass


def runGames(args):
    env = Wrap_pacman(args)
    return env


def runGames_2(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):

    rules = ClassicGameRules(timeout)
    games = []

    for i in range(numGames):
        beQuiet = i < numTraining

        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True

        # render
        # gameDisplay = display
        # rules.quiet = False

        game = rules.newGame(layout, pacman, ghosts,
                             gameDisplay, beQuiet, catchExceptions)
        game.run()

    return games


def make_env(args):
    #args = readCommand(sys.argv[1:])  # Get game components based on input
    #print(args)
    return runGames(args)
    # runGames_2(**args)
    # return env