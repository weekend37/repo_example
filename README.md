# repo_example

An example of how the submission repository should be structured

## Changes in BG_Competition.py from Backgammon.py

### Move Legitimacy
Added function for checking if the agent makes legit moves. 
Increases the running time a lot but necassery for the competition.

### Check if dice are the same before searching for possible single moves (minor change)
in the *legal_moves* function the possible moves are found. 
When there is only one possible move and the same number appears on both dice, it is useless to check possible moves for both dice. This should save a little time but the main reason for this change is to prevent confusing output in extreme cases that some teams were having trouble with.
