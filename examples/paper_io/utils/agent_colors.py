import random

# Expanded list of possible colors
# Expanded list of possible colors with names
color_name_mapping = {
    (0, 255, 0): "Green",
    (255, 0, 0): "Red",
    (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow",
    (255, 0, 255): "Magenta",
    (0, 255, 255): "Cyan",
    (128, 0, 128): "Purple",
    (255, 165, 0): "Orange",
    (139, 69, 19): "Brown",
    (75, 0, 130): "Indigo",
    (255, 192, 203): "Pink",
    (0, 128, 128): "Teal"
}

# List of possible colors for assignment
possible_colors = list(color_name_mapping.keys())  # This ensures we use the RGB tuples from the mapping

# Function to assign random colors to agents and return both RGB and name
def assign_agent_colors(num_players):
    selected_colors = random.sample(possible_colors, num_players)  # Pick random colors
    color_info = [(color, color_name_mapping[color]) for color in selected_colors]  # Get both RGB and name
    return color_info  # List of tuples [(RGB, name), ...]
