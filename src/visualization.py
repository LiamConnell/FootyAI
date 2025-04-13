import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import numpy as np
from typing import List
from PIL import Image

from src.game_state import GameState
from src.config import (
    FIELD_WIDTH,
    FIELD_HEIGHT,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    BALL_COLOR,
    FIELD_COLOR,
    FIELD_ALPHA,
    LINE_COLOR,
    PENALTY_WIDTH,
    PENALTY_HEIGHT,
    GOAL_WIDTH,
    GOAL_HEIGHT,
    CENTER_CIRCLE_RADIUS,
    SPOT_RADIUS,
    FIGURE_WIDTH,
    FIGURE_HEIGHT,
    FIGURE_DPI
)

def states_to_mp4(states: list[GameState], path: str, fps: int = 10):
    """Convert a list of game states to an MP4 video."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, FIELD_WIDTH )
    ax.set_ylim(0, FIELD_HEIGHT)
    ax.set_aspect('equal')
    ax.axis('off')
    
    _draw_field_on_ax(ax)
    
    def init():
        return []
    
    field_patches = len(ax.patches)
        
    
    def animate(frame_num):
        # Clear previous patches (except field elements)
        for patch in ax.patches[field_patches:]:  # Keep first 13 patches (field elements)
            patch.remove()
        
        state = states[frame_num]
        
        # Draw team A players
        for player in state.team_a:
            ax.add_patch(Circle(player.position, 2, color=TEAM_A_COLOR))
        
        # Draw team B players
        for player in state.team_b:
            ax.add_patch(Circle(player.position, 2, color=TEAM_B_COLOR))
        
        # Draw ball
        ax.add_patch(Circle(state.ball.position, 1, color=BALL_COLOR))
        
        # Add score and time
        ax.text(5, 5, f"Score: {state.score[0]} - {state.score[1]}", 
                fontsize=12, color='white', weight='bold')
        ax.text(FIELD_WIDTH-30, 5, f"Time: {state.time_remaining}", 
                fontsize=12, color='white', weight='bold')
        
        return ax.patches
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(states),
        init_func=init,
        interval=1000/fps,
        blit=True
    )
    
    # Save the animation
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(path, writer=writer, dpi=FIGURE_DPI)

    plt.close(fig)

def _draw_field_on_ax(ax):
    """Draw the soccer field on a given axis."""
    # Draw green field background
    ax.add_patch(Rectangle((0, 0), FIELD_WIDTH, FIELD_HEIGHT, 
                            fill=True, color=FIELD_COLOR, alpha=FIELD_ALPHA))
    
    # Draw field outline
    ax.add_patch(Rectangle((0, 0), FIELD_WIDTH, FIELD_HEIGHT, 
                            fill=False, color=LINE_COLOR, linewidth=2))
    
    # Draw center line
    ax.plot([FIELD_WIDTH/2, FIELD_WIDTH/2], [0, FIELD_HEIGHT], 
            LINE_COLOR, linewidth=1, linestyle='--')
    
    # Draw center circle
    center_circle = Circle((FIELD_WIDTH/2, FIELD_HEIGHT/2), CENTER_CIRCLE_RADIUS, 
                            fill=False, color=LINE_COLOR, linewidth=1)
    ax.add_patch(center_circle)
    
    # Draw penalty areas
    # Left penalty area
    ax.add_patch(Rectangle((0, (FIELD_HEIGHT-PENALTY_HEIGHT)/2), 
                            PENALTY_WIDTH, PENALTY_HEIGHT, 
                            fill=False, color=LINE_COLOR, linewidth=1))
    # Right penalty area
    ax.add_patch(Rectangle((FIELD_WIDTH-PENALTY_WIDTH, 
                            (FIELD_HEIGHT-PENALTY_HEIGHT)/2), 
                            PENALTY_WIDTH, PENALTY_HEIGHT, 
                            fill=False, color=LINE_COLOR, linewidth=1))
    
    # Draw goals
    # Left goal
    ax.add_patch(Rectangle((0, (FIELD_HEIGHT-GOAL_HEIGHT)/2), 
                            GOAL_WIDTH, GOAL_HEIGHT, 
                            fill=False, color=LINE_COLOR, linewidth=2))
    # Right goal
    ax.add_patch(Rectangle((FIELD_WIDTH-GOAL_WIDTH, 
                            (FIELD_HEIGHT-GOAL_HEIGHT)/2), 
                            GOAL_WIDTH, GOAL_HEIGHT, 
                            fill=False, color=LINE_COLOR, linewidth=2))
    
    # Add field markings
    # Center spot
    ax.add_patch(Circle((FIELD_WIDTH/2, FIELD_HEIGHT/2), SPOT_RADIUS, 
                        fill=True, color=LINE_COLOR))
    
    # Penalty spots
    ax.add_patch(Circle((PENALTY_WIDTH/2, FIELD_HEIGHT/2), SPOT_RADIUS, 
                        fill=True, color=LINE_COLOR))
    ax.add_patch(Circle((FIELD_WIDTH-PENALTY_WIDTH/2, FIELD_HEIGHT/2), SPOT_RADIUS, 
                        fill=True, color=LINE_COLOR))


        
    

    

# class SoccerVisualizer:
#     def __init__(self):
#         # Set up the figure and field dimensions
#         self.field_width = FIELD_WIDTH
#         self.field_height = FIELD_HEIGHT
#         self.team_a_color = TEAM_A_COLOR
#         self.team_b_color = TEAM_B_COLOR
#         self.ball_color = BALL_COLOR
        
#         # Create figure with fixed DPI and size
#         self.fig, self.ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
#         self.ax.set_xlim(0, self.field_width)
#         self.ax.set_ylim(0, self.field_height)
#         self.ax.set_aspect('equal')
#         self.ax.axis('off')
#         self._draw_field()
        
#         # Force the figure to be drawn
#         # self.fig.canvas.draw()
    
#     def _draw_field(self):
#         """Setup the soccer field."""
#         # Draw green field background
#         self.ax.add_patch(Rectangle((0, 0), self.field_width, self.field_height, 
#                                   fill=True, color=FIELD_COLOR, alpha=FIELD_ALPHA))
        
#         # Draw field outline
#         self.ax.add_patch(Rectangle((0, 0), self.field_width, self.field_height, 
#                                   fill=False, color=LINE_COLOR, linewidth=2))
        
#         # Draw center line
#         self.ax.plot([self.field_width/2, self.field_width/2], [0, self.field_height], 
#                     LINE_COLOR, linewidth=1, linestyle='--')
        
#         # Draw center circle
#         center_circle = Circle((self.field_width/2, self.field_height/2), CENTER_CIRCLE_RADIUS, 
#                              fill=False, color=LINE_COLOR, linewidth=1)
#         self.ax.add_patch(center_circle)
        
#         # Draw penalty areas
#         # Left penalty area
#         self.ax.add_patch(Rectangle((0, (self.field_height-PENALTY_HEIGHT)/2), 
#                                   PENALTY_WIDTH, PENALTY_HEIGHT, 
#                                   fill=False, color=LINE_COLOR, linewidth=1))
#         # Right penalty area
#         self.ax.add_patch(Rectangle((self.field_width-PENALTY_WIDTH, 
#                                    (self.field_height-PENALTY_HEIGHT)/2), 
#                                   PENALTY_WIDTH, PENALTY_HEIGHT, 
#                                   fill=False, color=LINE_COLOR, linewidth=1))
        
#         # Draw goals
#         # Left goal
#         self.ax.add_patch(Rectangle((0, (self.field_height-GOAL_HEIGHT)/2), 
#                                   GOAL_WIDTH, GOAL_HEIGHT, 
#                                   fill=False, color=LINE_COLOR, linewidth=2))
#         # Right goal
#         self.ax.add_patch(Rectangle((self.field_width-GOAL_WIDTH, 
#                                    (self.field_height-GOAL_HEIGHT)/2), 
#                                   GOAL_WIDTH, GOAL_HEIGHT, 
#                                   fill=False, color=LINE_COLOR, linewidth=2))
        
#         # Add field markings
#         # Center spot
#         self.ax.add_patch(Circle((self.field_width/2, self.field_height/2), SPOT_RADIUS, 
#                                fill=True, color=LINE_COLOR))
        
#         # Penalty spots
#         self.ax.add_patch(Circle((PENALTY_WIDTH/2, self.field_height/2), SPOT_RADIUS, 
#                                fill=True, color=LINE_COLOR))
#         self.ax.add_patch(Circle((self.field_width-PENALTY_WIDTH/2, self.field_height/2), SPOT_RADIUS, 
#                                fill=True, color=LINE_COLOR))
    
#     # def render_state(self, state):
#     #     """Render a single frame of the game state."""
#     #     # Clear previous patches (except field elements)
#     #     for patch in self.ax.patches[13:]:  # Keep first 13 patches (field elements)
#     #         patch.remove()
        
#     #     # Draw team A players
#     #     for player in state.team_a:
#     #         self.ax.add_patch(Circle(player.position, 2, color=self.team_a_color))
        
#     #     # Draw team B players
#     #     for player in state.team_b:
#     #         self.ax.add_patch(Circle(player.position, 2, color=self.team_b_color))
        
#     #     # Draw ball
#     #     self.ax.add_patch(Circle(state.ball.position, 1, color=self.ball_color))
        
#     #     # Add score and time
#     #     self.ax.text(5, 5, f"Score: {state.score[0]} - {state.score[1]}", 
#     #                 fontsize=12, color='white', weight='bold')
#     #     self.ax.text(self.field_width-30, 5, f"Time: {state.time_remaining}", 
#     #                 fontsize=12, color='white', weight='bold')
        
#     #     # Update the plot
#     #     self.fig.canvas.draw()
#     #     self.fig.canvas.flush_events()
        
#     #     return self.fig, self.ax
    
#     # def get_rgb_array(self, state):
#     #     """Get RGB array representation of the current state."""
#     #     self.render_state(state)
#     #     self.fig.canvas.draw()
        
#     #     # Get the buffer and reshape it
#     #     # data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
#     #     # width, height = self.fig.canvas.get_width_height()
#     #     # data = data.reshape((height, width, 3))

#     #     buffer = self.fig.canvas.tostring_argb()
#     #     width, height = self.fig.canvas.get_width_height()
#     #     width, height = width * 2, height * 2

#     #     # Convert to NumPy array
#     #     data = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))

#     #     # Convert ARGB to RGBA (Matplotlib sometimes gives ARGB instead of RGBA)
#     #     data = data[:, :, [1, 2, 3, 0]]

#     #     # Drop the alpha channel to get RGB only
#     #     rgb_data = data[:, :, :3]
        
#     #     # # Ensure the array is properly sized
#     #     # if data.shape != (700, 1000, 3):
#     #     #     # Resize to match expected dimensions
#     #     #     img = Image.fromarray(data)
#     #     #     img = img.resize((1000, 700))
#     #     #     data = np.array(img)
        
#     #     return rgb_data
    
#     # def close(self):
#     #     """Close the visualization window."""
#     #     plt.close(self.fig)
        
#     def save_game_sequence(self, states: List[GameState], filename: str, fps: int = 30, show: bool = False):
#         """Save a sequence of game states as an MP4 video."""
#         # Create a new figure for the animation
#         fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
#         ax.set_xlim(0, self.field_width)
#         ax.set_ylim(0, self.field_height)
#         ax.set_aspect('equal')
#         ax.axis('off')
        
#         # Draw the field once
#         self._draw_field_on_ax(ax)

#         field_patches = len(ax.patches)
        
#         def init():
#             return []
        
#         def animate(frame_num):
#             # Clear previous patches (except field elements)
#             for patch in ax.patches[field_patches:]:  # Keep first 13 patches (field elements)
#                 patch.remove()
            
#             state = states[frame_num]
            
#             # Draw team A players
#             for player in state.team_a:
#                 ax.add_patch(Circle(player.position, 2, color=self.team_a_color))
            
#             # Draw team B players
#             for player in state.team_b:
#                 ax.add_patch(Circle(player.position, 2, color=self.team_b_color))
            
#             # Draw ball
#             ax.add_patch(Circle(state.ball.position, 1, color=self.ball_color))
            
#             # Add score and time
#             ax.text(5, 5, f"Score: {state.score[0]} - {state.score[1]}", 
#                    fontsize=12, color='white', weight='bold')
#             ax.text(self.field_width-30, 5, f"Time: {state.time_remaining}", 
#                    fontsize=12, color='white', weight='bold')
            
#             return ax.patches
        
#         # Create the animation
#         anim = animation.FuncAnimation(
#             fig, animate, frames=len(states),
#             init_func=init,
#             interval=1000/fps,
#             blit=True
#         )
        
#         # Save the animation
#         writer = animation.FFMpegWriter(fps=fps)
#         anim.save(filename, writer=writer, dpi=FIGURE_DPI)
        
#         if show:
#             plt.show()
#         else:
#             plt.close(fig)
    
#     def _draw_field_on_ax(self, ax):
#         """Draw the soccer field on a given axis."""
#         # Draw green field background
#         ax.add_patch(Rectangle((0, 0), self.field_width, self.field_height, 
#                              fill=True, color=FIELD_COLOR, alpha=FIELD_ALPHA))
        
#         # Draw field outline
#         ax.add_patch(Rectangle((0, 0), self.field_width, self.field_height, 
#                              fill=False, color=LINE_COLOR, linewidth=2))
        
#         # Draw center line
#         ax.plot([self.field_width/2, self.field_width/2], [0, self.field_height], 
#                LINE_COLOR, linewidth=1, linestyle='--')
        
#         # Draw center circle
#         center_circle = Circle((self.field_width/2, self.field_height/2), CENTER_CIRCLE_RADIUS, 
#                              fill=False, color=LINE_COLOR, linewidth=1)
#         ax.add_patch(center_circle)
        
#         # Draw penalty areas
#         # Left penalty area
#         ax.add_patch(Rectangle((0, (self.field_height-PENALTY_HEIGHT)/2), 
#                              PENALTY_WIDTH, PENALTY_HEIGHT, 
#                              fill=False, color=LINE_COLOR, linewidth=1))
#         # Right penalty area
#         ax.add_patch(Rectangle((self.field_width-PENALTY_WIDTH, 
#                               (self.field_height-PENALTY_HEIGHT)/2), 
#                              PENALTY_WIDTH, PENALTY_HEIGHT, 
#                              fill=False, color=LINE_COLOR, linewidth=1))
        
#         # Draw goals
#         # Left goal
#         ax.add_patch(Rectangle((0, (self.field_height-GOAL_HEIGHT)/2), 
#                              GOAL_WIDTH, GOAL_HEIGHT, 
#                              fill=False, color=LINE_COLOR, linewidth=2))
#         # Right goal
#         ax.add_patch(Rectangle((self.field_width-GOAL_WIDTH, 
#                               (self.field_height-GOAL_HEIGHT)/2), 
#                              GOAL_WIDTH, GOAL_HEIGHT, 
#                              fill=False, color=LINE_COLOR, linewidth=2))
        
#         # Add field markings
#         # Center spot
#         ax.add_patch(Circle((self.field_width/2, self.field_height/2), SPOT_RADIUS, 
#                           fill=True, color=LINE_COLOR))
        
#         # Penalty spots
#         ax.add_patch(Circle((PENALTY_WIDTH/2, self.field_height/2), SPOT_RADIUS, 
#                           fill=True, color=LINE_COLOR))
#         ax.add_patch(Circle((self.field_width-PENALTY_WIDTH/2, self.field_height/2), SPOT_RADIUS, 
#                           fill=True, color=LINE_COLOR))
    
#     def show(self):
#         """Display the current figure."""
#         plt.show() 