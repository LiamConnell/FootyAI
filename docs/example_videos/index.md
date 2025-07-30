# Example Videos

## Typical Start State

When the model is initialized, the players move around randomly. In many cases, they will simply drift into the corners as the game state has almost no change. 

<div style="text-align: center;">
    <video width="400" controls>
        <source src="./1_typical_initial_state.mp4" type="video/mp4">
    </video>
</div>

## Full training run with poor reward function

The players slowly advance to the ball and kick it gently toward the goal. I think this behavior stems from the reward function that I defined for this run. Players are rewarded if their distance to the ball is decreased and if they kick the ball. This encourages them to move directly to the ball and discourages them from kicking the ball hard, which would increase the distance to the ball and make it less likely for them to kick it again. 

<div style="text-align: center;">
    <video width="400" controls>
        <source src="./2025-07-29_charge.mp4" type="video/mp4">
    </video>
</div>