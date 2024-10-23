# Visual Navigation Game

This is the course project platform for NYU ROB-GY 6203 Robot Perception. 
For more information, please reach out to AI4CE lab (cfeng at nyu dot edu).

# Instructions for Players
1. Install
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
```

2. Play using the default keyboard player
```commandline
python player.py
```

3. Modify the player.py to implement your own solutions, 
unless you have photographic memories!

# Instructions for Judges

## Install
Notice the repository difference, players have no access to vis_nav_game.git
```console
conda update conda
git clone https://github.com/ai4ce/vis_nav_game.git
cd vis_nav_game/
conda env create -f environment.yaml
conda activate game
```

For all following, make sure your current directory is inside the main vis_nav_game.git folder

## Generate the new maze and target pose
```console
python maze.py
```

## Run Test for Judges
```console
python vis_nav_core.py
```

## View Player-submitted game.npy
```commandline
python vis_nav_core.py -m judge -i 0 -f game.npy -s 1
```
