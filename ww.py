from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import imageio

sourcedir = 'D:/NEU/ImageRestoration/div2k/Set14'
names_hr = []
gif_name = 'urban_2.gif'
duration = 0.5
for dirpath, _, fnames in sorted(os.walk(sourcedir)):
    for fname in sorted(fnames):
        names_hr.append(imageio.imread(os.path.join(sourcedir,fname)))
        imageio.mimsave(gif_name, names_hr, 'GIF', duration=duration)