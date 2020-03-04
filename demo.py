import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as an

"""
# MorFE

Features:
 - predict sample class
 - predict cell organization
 - show training animation (w loss + metrics plots)

"""


fps = 30
nSeconds = 5
snapshots = [np.random.rand(5, 5) for _ in range(nSeconds * fps)]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8, 8))

a = snapshots[0]
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)


def animate_func(i):
    if i % fps == 0:
        print('.', end='')

    im.set_array(snapshots[i])
    return [im]


anim = an.FuncAnimation(
    fig,
    animate_func,
    frames=nSeconds * fps,
    interval=1000 / fps,  # in ms
)

anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

print('Done!')
