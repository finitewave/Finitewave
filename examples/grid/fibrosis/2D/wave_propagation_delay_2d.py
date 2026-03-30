"""
Propagation Delay Due to Diffuse Fibrosis
=========================================

This example compares wave propagation in two 2D cardiac tissues:
one healthy (no fibrosis) and one with 20% diffuse fibrosis.

A planar wave stimulus is applied from the left side of the tissue,
and propagation is simulated using the Aliev-Panfilov model.

The resulting transmembrane potential maps clearly show how diffuse
fibrosis slows down the conduction, causing a visible delay in
activation front propagation compared to the healthy tissue.

Setup:
------
- Tissue size: 300 × 300
- Fibrosis type: Diffuse (random spatial blockage)
- Fibrosis density: 20% (in fibrotic case)
- Stimulus:
    • Type: Voltage
    • Applied on leftmost 5 columns of the tissue
    • Time: t = 0 ms
- Model: Aliev-Panfilov 2D
- Time window: 20 ms

Visualization:
--------------
The resulting `u` (voltage) maps are plotted side by side to highlight
the delayed wavefront in the fibrotic tissue.

"""
import matplotlib.pyplot as plt
import finitewave as fw

n = 300
stim_x1, stim_x2 = 0, 5  # planar stimulus strip

def setup_tissue(with_fibrosis):
    tissue = fw.CardiacTissue2D((n, n))
    if with_fibrosis:
        tissue.add_pattern(fw.Diffuse2DPattern(density=0.2))
    return tissue

def run_simulation(tissue):
    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1,
                                                 x1=stim_x1, x2=stim_x2,
                                                 y1=0, y2=n))
    
    model = fw.AlievPanfilov2D()
    model.dt = 0.01
    model.dr = 0.25
    model.t_max = 20
    model.cardiac_tissue = tissue
    model.stim_sequence = stim_sequence
    model.run()
    return model

# Run simulations
print("Running healthy tissue...")
healthy_tissue = setup_tissue(with_fibrosis=False)
healthy_model = run_simulation(healthy_tissue)

print("Running fibrotic tissue (20% diffuse)...")
fibrotic_tissue = setup_tissue(with_fibrosis=True)
fibrotic_model = run_simulation(fibrotic_tissue)

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(healthy_model.u, cmap="viridis", origin="lower")
axs[0].set_title("Healthy Tissue (No Fibrosis)")
axs[0].axis("off")

axs[1].imshow(fibrotic_model.u, cmap="viridis", origin="lower")
axs[1].set_title("Diffuse Fibrosis (20%)")
axs[1].axis("off")

fig.suptitle("Propagation Delay Due to Fibrosis", fontsize=16)
plt.tight_layout()
plt.show()
