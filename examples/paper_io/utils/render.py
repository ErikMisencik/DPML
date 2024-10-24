import pygame
import numpy as np
def render_game(screen, grid, players, alive, cell_size, window_size, num_players, steps, agent_colors):
    
    # Initialize pygame fontsplayer_colors
    pygame.font.init()
    font = pygame.font.SysFont(None, 35)  # Font definition inside the function

    # Fill background with white
    screen.fill((230, 230, 230))  # Light grey background

    # Draw the circular arena with a beveled effect for 3D
    center = (window_size // 2, window_size // 2)
    radius = window_size // 2 - 20

    # Beveled border effect for the arena
    pygame.draw.circle(screen, (200, 200, 200), center, radius + 20)  # Outer darker ring
    pygame.draw.circle(screen, (255, 255, 255), center, radius)  # White inner circle (arena)

    # Draw trails and territories on the arena surface
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            cell_value = grid[x, y]
            top_left = (y * cell_size, x * cell_size)
            rect = pygame.Rect(top_left[0], top_left[1], cell_size, cell_size)

            if cell_value > 0:
                # Territory with subtle 3D effect
                player_id = cell_value - 1
                pygame.draw.rect(screen, agent_colors[player_id], rect)

                # Add subtle 3D shading for territories
                light_color = [min(255, int(c * 1.05)) for c in agent_colors[player_id]]
                shadow_color = [max(0, int(c * 0.9)) for c in agent_colors[player_id]]
                pygame.draw.line(screen, light_color, rect.topleft, (rect.right, rect.top), 1)
                pygame.draw.line(screen, light_color, rect.topleft, (rect.left, rect.bottom), 1)
                pygame.draw.line(screen, shadow_color, rect.bottomright, (rect.right, rect.top), 1)
                pygame.draw.line(screen, shadow_color, rect.bottomright, (rect.left, rect.bottom), 1)

            elif cell_value < 0:
                # Trail with subtle 3D effect
                player_id = -cell_value - 1
                faded_color = [int(0.5 * 255 + 0.5 * c) for c in agent_colors[player_id]]
                pygame.draw.rect(screen, faded_color, rect)

                # Add subtle 3D shading for trails
                light_color = [min(255, int(c * 1.05)) for c in faded_color]
                shadow_color = [max(0, int(c * 0.9)) for c in faded_color]
                pygame.draw.line(screen, light_color, rect.topleft, (rect.right, rect.top), 1)
                pygame.draw.line(screen, light_color, rect.topleft, (rect.left, rect.bottom), 1)
                pygame.draw.line(screen, shadow_color, rect.bottomright, (rect.right, rect.top), 1)
                pygame.draw.line(screen, shadow_color, rect.bottomright, (rect.left, rect.bottom), 1)

    # Highlight players with a stronger 3D effect
    for i, player in enumerate(players):
        if not alive[i]:
            continue
        x, y = player['position']
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        color = [min(255, c + 100) for c in agent_colors[i]]

        # Draw player with a stronger 3D effect
        pygame.draw.rect(screen, color, rect)

        # Stronger highlight on the top-left to simulate light source for player
        light_color = [min(255, int(c * 1.3)) for c in color]
        pygame.draw.line(screen, light_color, rect.topleft, (rect.right, rect.top), 2)
        pygame.draw.line(screen, light_color, rect.topleft, (rect.left, rect.bottom), 2)

        # Stronger shadow on the bottom-right to simulate depth for player
        shadow_color = [max(0, int(c * 0.6)) for c in color]
        pygame.draw.line(screen, shadow_color, rect.bottomright, (rect.right, rect.top), 2)
        pygame.draw.line(screen, shadow_color, rect.bottomright, (rect.left, rect.bottom), 2)

    # Display the number of steps in the top-left corner
    step_text = font.render(f'Steps: {steps}', True, (0, 0, 0))  # Black text
    screen.blit(step_text, (10, 10))  # Render in the top-left corner

    # Update display
    pygame.display.flip()
