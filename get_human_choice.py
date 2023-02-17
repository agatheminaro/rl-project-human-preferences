import imageio
from IPython.display import display, clear_output
from ipywidgets import widgets
import numpy as np


def get_human_choice(obs_1, obs_2):
    obs_1 = (np.array(obs_1) * 255).astype(np.uint8)
    obs_2 = (np.array(obs_2) * 255).astype(np.uint8)

    imageio.mimsave(
        "./example1.gif", obs_1, fps=15  # output gif  # array of input frames
    )

    imageio.mimsave("./example2.gif", obs_2, fps=15)

    clear_output(True)

    print("Choose between 2 scenarios:")
    print(
        "0: They are equally good, 1: The 1st is better, 2: The 2nd is better, 3: None are good"
    )

    img1 = open("example1.gif", "rb").read()
    img2 = open("example2.gif", "rb").read()

    wi1 = widgets.Image(value=img1, format="png", width=300, height=400)
    wi2 = widgets.Image(value=img2, format="png", width=300, height=400)

    sidebyside = widgets.HBox([wi1, wi2])

    display(sidebyside)

    choice = input("Press 0 or 1 or 2 or 3:")

    if choice == 0:
        human_choice = [0.5, 0.5]
    elif choice == 1:
        human_choice = [1, 0]
    elif choice == 2:
        human_choice = [0, 1]
    else:
        human_choice = [0, 0]

    return human_choice
