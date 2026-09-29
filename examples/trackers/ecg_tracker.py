"""Compare three ways to record the same ECG on a 2D tissue grid.

ECGTracker computes inverse-distance weights at each sample.
LeadFieldECGTracker can precompute those weights from electrode coordinates,
or accept supplied fields with shape (number of leads, number of tissue nodes).
The three recordings should agree up to floating-point rounding.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw


def compute_lead_fields(tissue, lead_coords, volume_conductivity, dr):
    """Compute lead fields from electrode coordinates."""
    nodes = np.pad(tissue.tissue_coords, ((0, 0), (0, 1))) * tissue.dr
    electrodes = lead_coords * tissue.dr
    lead_fields = np.empty((len(electrodes), len(nodes)))
    for i, electrode in enumerate(electrodes):
        distances = np.linalg.norm(nodes - electrode, axis=1)
        lead_fields[i] = (dr ** 3) / (4 * np.pi * volume_conductivity * distances)
    return lead_fields


def main(t_max=50, show=True):
    n, m = 300, 50
    dr = 0.1
    tissue = fw.CardiacTissue(shape=(n, m), dr=dr)
    lead_coords = np.array([[5, m//2, 5],
                            [n//2, m//2, 5],
                            [n//2, 5, 5]])
    volume_conductivity = 1.0

    # Grid electrode coordinates are in grid units. Supplied fields must use
    # physical distances and include the desired conductivity normalization.
    # Columns follow tissue.tissue_coords, matching the compact voltage vector.
    lead_fields = compute_lead_fields(tissue, lead_coords, volume_conductivity, dr)

    onfly = fw.ECGTracker(lead_coords, volume_conductivity, step=10)
    precomputed = fw.LeadFieldECGTracker(lead_coords,
                                         volume_conductivity=volume_conductivity,
                                         step=10)
    supplied = fw.LeadFieldECGTracker(lead_fields=lead_fields, step=10)

    tracker_sequence = fw.TrackerSequence()
    tracker_sequence.add_tracker(onfly)
    tracker_sequence.add_tracker(precomputed)
    tracker_sequence.add_tracker(supplied)

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimVoltageCoord(0, 1, 0, 5, 0, m))

    simulation = fw.CardiacSimulation(dt=0.01, t_max=t_max, backend="jax")
    simulation.cardiac_tissue = tissue
    simulation.cardiac_model = fw.LuoRudy91()
    simulation.stim_sequence = stim_sequence
    simulation.tracker_sequence = tracker_sequence
    simulation.run()

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['On-fly calculation', 'Precomputed from coordinates', 'Supplied fields']
    styles = ['-', '--', ':']
    fig, axs = plt.subplots(1, 4, figsize=(15, 4),
                            gridspec_kw={"width_ratios": [0.6, 1, 1, 1]})
    axs[0].imshow(simulation.cardiac_model.u, origin='lower')
    axs[0].set_title("Tissue and electrodes")
    for i, coord in enumerate(lead_coords):
        axs[0].scatter(coord[1], coord[0], color=colors[i])
        axs[0].annotate(f'Electrode {i}', (coord[1], coord[0]), color=colors[i], fontsize=10)

        for label, tracker, style in zip(labels, [onfly, precomputed, supplied], styles):
            signals = np.asarray(tracker.ecg).reshape(-1, len(lead_coords))
            axs[i + 1].plot(tracker.tracking_times, signals[:, i], style,
                            label=label)
        axs[i + 1].set_title(f"Electrode {i}")
        axs[i + 1].set_xlabel("Time (ms)")
        axs[i + 1].set_ylabel("ECG potential (model units)")
        axs[i + 1].sharex(axs[2])
        axs[i + 1].sharey(axs[2])

    axs[-1].legend()
    fig.tight_layout()
    if show:
        plt.show()
    return simulation, fig


if __name__ == "__main__":
    main()
