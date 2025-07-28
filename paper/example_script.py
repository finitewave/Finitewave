import finitewave as fw

import numpy as np
import matplotlib.pyplot as plt


n = 200
tissue = fw.CardiacTissue2D([n, n])

aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 120

stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0,
                                             volt_value=1,
                                             x1=1, x2=n-1, y1=1, y2=n//2))
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=31,
                       volt_value=1,
                       x1=1, x2=n//2, y1=1, y2=n-1))

aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

plt.imshow(aliev_panfilov.u, cmap='Spectral_r')
plt.axis('off')
plt.show()