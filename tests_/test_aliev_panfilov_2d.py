import sys
import unittest
import numpy as np

import finitewave as fw


class TestAlievPanfilov2D(unittest.TestCase):
    def setUp(self):

        n = 200
        self.tissue = fw.CardiacTissue2D([n, n])
        self.tissue.mesh = np.ones([n, n], dtype="uint8")
        self.tissue.add_boundaries()
        self.tissue.fibers = np.zeros([n, n, 2])
        self.tissue.stencil = fw.AsymmetricStencil2D()

        self.aliev_panfilov = fw.AlievPanfilov2D()
        self.aliev_panfilov.dt = 0.01
        self.aliev_panfilov.dr = 0.25
        self.aliev_panfilov.t_max = 25

        stim_sequence = fw.StimSequence()
        stim_sequence.add_stim(fw.StimCurrentCoord2D(0, 3, 0.18, 0, 200, 0, 5))

        tracker_sequence = fw.TrackerSequence()
        self.velocity_tracker = fw.Velocity2DTracker()
        self.velocity_tracker.threshold = 0.2
        tracker_sequence.add_tracker(self.velocity_tracker)

        self.aliev_panfilov.cardiac_tissue = self.tissue
        self.aliev_panfilov.stim_sequence = stim_sequence
        self.aliev_panfilov.tracker_sequence = tracker_sequence

    def test_wave_along_the_fibers(self):
        sys.stdout.write("---> Check the wave speed along the fibers\n")
        self.tissue.fibers[:, :, 0] = 0.
        self.tissue.fibers[:, :, 1] = 1.

        self.aliev_panfilov.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 1.6,
                               msg="Wave velocity along the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)

    def test_wave_across_the_fibers(self):
        sys.stdout.write("---> Check the wave speed across the fibers\n")
        self.tissue.fibers[:, :, 0] = 1.
        self.tissue.fibers[:, :, 1] = 0.
        self.tissue.D_al = 1
        self.tissue.D_ac = 0.111

        self.aliev_panfilov.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 0.55,
                               msg="Wave velocity across the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)

    def test_spiral_wave_period(self):
        sys.stdout.write("---> Check the spiral wave period\n")
        self.tissue.fibers[:, :, 0] = 0.
        self.tissue.fibers[:, :, 1] = 1.
        self.tissue.D_al = 1.
        self.tissue.D_ac = 1.
        self.aliev_panfilov.t_max = 200

        stim_sequence = fw.StimSequence()
        stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 200, 0, 100))
        stim_sequence.add_stim(fw.StimVoltageCoord2D(31, 1, 0, 100, 0, 200))

        tracker_sequence = fw.TrackerSequence()
        period_tracker = fw.Period2DTracker()
        period_tracker.cell_ind = [100, 100]
        period_tracker.threshold = 0.2
        tracker_sequence.add_tracker(period_tracker)

        spiral_tracker = fw.Spiral2DTracker()
        tracker_sequence.add_tracker(spiral_tracker)

        self.aliev_panfilov.stim_sequence = stim_sequence
        self.aliev_panfilov.tracker_sequence = tracker_sequence

        self.aliev_panfilov.run()

        self.assertAlmostEqual(period_tracker.output["100,100"][-1][1], 25.8,
                               msg="Spiral wave period is incorrect! (AlievPanfilov 2D)",
                               delta=0.3)

        spiral_tracker.write()
