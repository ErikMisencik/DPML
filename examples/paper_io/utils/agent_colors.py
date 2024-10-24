import random

# Expanded list of possible colors
possible_colors = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (139, 69, 19),  # Brown
    (75, 0, 130),   # Indigo
    (255, 192, 203),# Pink
    (0, 128, 128)   # Teal
]

def assign_agent_colors(num_agents):
    """
    Assign random colors to agents from the possible colors list.
    Ensures that each agent gets a unique color.
    
    :param num_agents: Number of agents to assign colors to
    :return: List of colors corresponding to each agent
    """
    if num_agents > len(possible_colors):
        raise ValueError(f"Too many agents! Only {len(possible_colors)} unique colors available.")
    
    # Randomly select unique colors for the agents
    return random.sample(possible_colors, num_agents)
