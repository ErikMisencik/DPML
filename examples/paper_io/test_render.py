# test_render.py
import cv2
from examples.paper_io.Paper_io import PaperIoEnv
# from make_env import make_env

def main():
    env = PaperIoEnv()
    obs = env.reset()

    done = False
    while not done:
        actions = []
        for i in range(env.num_players):
            if env.alive[i]:
                actions.append(env.action_space.sample())
            else:
                actions.append(None)  # Placeholder for eliminated players

        obs, rewards, done, info = env.step(actions)
        env.render()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
