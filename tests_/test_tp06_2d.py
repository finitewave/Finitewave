import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys

import finitewave as fw


class TestTP062D(unittest.TestCase):
    def setUp(self):

        n = 200
        self.tissue = fw.CardiacTissue2D([n, n])
        self.tissue.mesh = np.ones([n, n], dtype="uint8")
        self.tissue.add_boundaries()
        self.tissue.fibers = np.zeros([n, n, 2])
        self.tissue.stencil = fw.AsymmetricStencil2D()
        self.tissue.D_al  = 0.154
        self.tissue.D_ac  = 0.154

        self.tp06 = fw.TP062D()
        self.tp06.dt    = 0.001
        self.tp06.dr    = 0.1
        self.tp06.t_max = 10

        stim_sequence = fw.StimSequence()
        stim_sequence.add_stim(fw.StimVoltageCoord2D(0, -20, 0, 200, 0, 5))

        tracker_sequence = fw.TrackerSequence()
        self.velocity_tracker = fw.Velocity2DTracker()
        self.velocity_tracker.threshold = -60
        tracker_sequence.add_tracker(self.velocity_tracker)

        self.tp06.cardiac_tissue   = self.tissue
        self.tp06.stim_sequence    = stim_sequence
        self.tp06.tracker_sequence = tracker_sequence

    def test_wave_along_the_fibers(self):
        sys.stdout.write("---> Check the wave speed along the fibers\n")
        self.tissue.fibers[:,:,0] = 0.
        self.tissue.fibers[:,:,1] = 1.

        stim_sequence = fw.StimSequence()
        stim_sequence.add_stim(fw.StimVoltageCoord2D(0, -20, 0, 200, 0, 5))
        self.tp06.stim_sequence  = stim_sequence

        self.tp06.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())

        self.assertAlmostEqual(front_vel, 0.8,
                               msg="Wave velocity along the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)


    def test_wave_across_the_fibers(self):
        sys.stdout.write("---> Check the wave speed across the fibers\n")
        self.tissue.fibers[:,:,0] = 1.
        self.tissue.fibers[:,:,1] = 0.
        self.tissue.D_al = 0.154
        self.tissue.D_ac = 0.0171

        stim_sequence = fw.StimSequence()
        stim_sequence.add_stim(fw.StimVoltageCoord2D(0, -20, 0, 200, 0, 5))
        self.tp06.stim_sequence  = stim_sequence

        self.tp06.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 0.23,
                               msg="Wave velocity across the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)

    # def test_spiral_wave_period(self):
    #     sys.stdout.write("---> Check the spiral wave period\n")
    #    self.tissue.fibers[:,:,0] = 1.
    #    self.tissue.fibers[:,:,1] = 0.
    #    self.tissue.D_al = 0.154
    #    self.tissue.D_ac = 0.154
    #
    #     stim_params = [[0, 200, 0, 100, -20, 0.],
    #                    [0, 100, 0, 200, -20, 31.]]
    #     self.tp06.cardiac_tissue = self.tissue
    #     self.tp06.stim_params    = stim_params
    #
    #     period_tracker = Period2DTracker()
    #     period_tracker.target_model = self.tp06
    #     period_tracker.mode = "Detectors"
    #
    #     detectors = np.zeros([200, 200], dtype="uint8")
    #     positions = np.array([[100, 100]])
    #     detectors[positions[:, 0], positions[:, 1]] = 1
    #
    #     period_tracker.detectors = detectors
    #     period_tracker.threshold = 0.2
    #
    #     # add tracker to the model
    #     self.tp06.add_tracker(period_tracker)
    #
    #     self.tp06.run()
    #
    #     self.assertAlmostEqual(period_tracker.output["100,100"][-1][1], 194.,
    #                            msg="Spiral wave period is incorrect! (AlievPanfilov 2D)",
    #                            delta=1.)
