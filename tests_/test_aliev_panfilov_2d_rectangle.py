import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw


class TestAlievPanfilov2DRectangle(unittest.TestCase):
    def setUp(self):

        n_i = 100
        n_j = 300
        self.tissue = fw.CardiacTissue2D([n_i, n_j])
        self.tissue.mesh = np.ones([n_i, n_j], dtype="uint8")
        self.tissue.add_boundaries()
        self.tissue.fibers = np.zeros([n_i, n_j, 2])
        self.tissue.stencil = fw.AsymmetricStencil2D()

        self.aliev_panfilov = fw.AlievPanfilov2D()
        self.aliev_panfilov.dt    = 0.01
        self.aliev_panfilov.dr    = 0.25
        self.aliev_panfilov.t_max = 25

        stim_sequence = fw.StimSequence()
        stim_sequence.add_stim(fw.StimCurrentCoord2D(0, 3, 0.18, 0, 100, 0, 5))

        tracker_sequence = fw.TrackerSequence()
        self.velocity_tracker = fw.Velocity2DTracker()
        self.velocity_tracker.threshold = 0.2
        tracker_sequence.add_tracker(self.velocity_tracker)

        self.aliev_panfilov.cardiac_tissue   = self.tissue
        self.aliev_panfilov.stim_sequence    = stim_sequence
        self.aliev_panfilov.tracker_sequence = tracker_sequence

    def test_wave_along_the_fibers(self):
        sys.stdout.write("---> Check the wave speed along the fibers\n")
        self.tissue.fibers[:,:,0] = 0.
        self.tissue.fibers[:,:,1] = 1.

        self.aliev_panfilov.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 1.6,
                               msg="Wave velocity along the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)

    def test_wave_across_the_fibers(self):
        sys.stdout.write("---> Check the wave speed across the fibers\n")
        self.tissue.fibers[:,:,0] = 1.
        self.tissue.fibers[:,:,1] = 0.
        self.tissue.D_al = 1
        self.tissue.D_ac = 0.111

        self.aliev_panfilov.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 0.55,
                               msg="Wave velocity across the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)
