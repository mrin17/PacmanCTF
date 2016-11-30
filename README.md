# PacmanCTF
Pacman Capture The Flag project for CS4100 AI

Calvin Pomerantz, Lyn Kotuby, Michael Rinaldi

### Playing A Game After Training (9 rounds of training, 1 game)
```python capture.py -r qLearningTeam -b qLearningTeam -x 9 -n 10```

### How to play game:
- By default, you can run a game with the simple baselineTeam that the staff has provided:
```python capture.py```
- A wealth of options are available to you:
```python capture.py --help```
- There are four slots for agents, where agents 0 and 2 are always on the red team, and 1 and 3 are on the blue team. Agents are created by agent factories (one for Red, one for Blue). See the section on designing agents for a description of the agents invoked above. The only team that we provide is the baselineTeam. It is chosen by default as both the red and blue team, but as an example of how to choose teams:
```python capture.py -r baselineTeam -b baselineTeam```
- which specifies that the red team -r and the blue team -b are both created from baselineTeam.py. To control one of the four agents with the keyboard, pass the appropriate option:
```python capture.py --keys0```
- The arrow keys control your character, which will change from ghost to Pacman when crossing the center line.

### Playing Multiple Games at Once Without Graphics
```python capture.py -r baselineTeam -b baselineTeam -q -n 10```

### Layouts
- By default, all games are run on the defaultcapture layout. To test your agent on other layouts, use the -l option. In particular, you can generate random layouts by specifying RANDOM[seed]. For example, -l RANDOM13 will use a map randomly generated with seed 13.

### Recordings
- You can record local games using the --record option, which will write the game history to a file named by the time the game was played. You can replay these histories using the --replay option and specifying the file to replay. All online matches are automatically recorded and the most recent ones can be viewed on the contest site. You are also able to download the history associated with each replay.

### CAPTURE HELP
```
Options:
  -h, --help            show this help message and exit
  -r RED, --red=RED     Red team [Default: baselineTeam]
  -b BLUE, --blue=BLUE  Blue team [Default: baselineTeam]
  --red-name=RED_NAME   Red team name [Default: Red]
  --blue-name=BLUE_NAME
                        Blue team name [Default: Blue]
  --redOpts=REDOPTS     Options for red team (e.g. first=keys) [Default: ]
  --blueOpts=BLUEOPTS   Options for blue team (e.g. first=keys) [Default: ]
  --keys0               Make agent 0 (first red player) a keyboard agent
  --keys1               Make agent 1 (second red player) a keyboard agent
  --keys2               Make agent 2 (first blue player) a keyboard agent
  --keys3               Make agent 3 (second blue player) a keyboard agent
  -l LAYOUT_FILE, --layout=LAYOUT_FILE
                        the LAYOUT_FILE from which to load the map layout; use
                        RANDOM for a random maze; use RANDOM<seed> to use a
                        specified random seed, e.g., RANDOM23 [Default:
                        defaultCapture]
  -t, --textgraphics    Display output as text only
  -q, --quiet           Display minimal output and no graphics
  -Q, --super-quiet     Same as -q but agent output is also suppressed
  -z ZOOM, --zoom=ZOOM  Zoom in the graphics [Default: 1]
  -i TIME, --time=TIME  TIME limit of a game in moves [Default: 1200]
  -n NUMGAMES, --numGames=NUMGAMES
                        Number of games to play [Default: 1]
  -f, --fixRandomSeed   Fixes the random seed to always play the same game
  --record              Writes game histories to a file (named by the time
                        they were played)
  --replay=REPLAY       Replays a recorded game file.
  -x NUMTRAINING, --numTraining=NUMTRAINING
                        How many episodes are training (suppresses output)
                        [Default: 0]
  -c, --catchExceptions
                        Catch exceptions and enforce time limits
```