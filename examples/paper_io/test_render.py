import cv2

from examples.paper_io.test import PaperIoEnv


def main():
    env = PaperIoEnv(num_players=2)
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
