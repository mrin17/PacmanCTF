# PacmanCTF
Pacman Capture The Flag project for CS4100 AI

Calvin Pomerantz, Lyn Kotuby, Michael Rinaldi


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

### Layouts
- By default, all games are run on the defaultcapture layout. To test your agent on other layouts, use the -l option. In particular, you can generate random layouts by specifying RANDOM[seed]. For example, -l RANDOM13 will use a map randomly generated with seed 13.

### Recordings
- You can record local games using the --record option, which will write the game history to a file named by the time the game was played. You can replay these histories using the --replay option and specifying the file to replay. All online matches are automatically recorded and the most recent ones can be viewed on the contest site. You are also able to download the history associated with each replay.