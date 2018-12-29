The folder contains the agent.py file which is standalone file to operate agent. The file contains Agent class which is build upon the interface provided and can be used to play Pong game.

The final-model file is the final trained model that could be used to test the agent.

Apart from final model there are also some interesting models in saved_models folder.
1. checkpoint-defensive is an overfitted model for simple-ai. It play long games with the opponent and have high win probability
2. checkpoint-offensive is a generic model that could be used to play against general opponents but it has low win probability.

The policy gradient folder contains training performed using policy gradient technique

