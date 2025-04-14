import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys

import finitewave as fw


class TestLR912D(unittest.TestCase):
    def setUp(self):

        n = 200
        self.tissue = fw.CardiacTissue2D([n, n])
        self.tissue.mesh = np.ones([n, n], dtype="uint8")
        self.tissue.add_boundaries()
        self.tissue.fibers = np.zeros([n, n, 2])
        self.tissue.stencil = fw.AsymmetricStencil2D()
        self.tissue.D_al  = 0.1
        self.tissue.D_ac  = 0.1

        self.lr91 = fw.LuoRudy912D()
        self.lr91.dt    = 0.001
        self.lr91.dr    = 0.1
        self.lr91.t_max = 10

        stim_sequence = fw.StimSequence()
        stim_sequence.add_stim(fw.StimVoltageCoord2D(0, -20, 0, 200, 0, 5))

        tracker_sequence = fw.TrackerSequence()
        self.velocity_tracker = fw.Velocity2DTracker()
        self.velocity_tracker.threshold = -60
        tracker_sequence.add_tracker(self.velocity_tracker)

        self.lr91.cardiac_tissue   = self.tissue
        self.lr91.stim_sequence    = stim_sequence
        self.lr91.tracker_sequence = tracker_sequence

    def test_wave_along_the_fibers(self):
        sys.stdout.write("---> Check the wave speed along the fibers\n")
        self.tissue.fibers[:,:,0] = 0.
        self.tissue.fibers[:,:,1] = 1.

        self.lr91.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())

        self.assertAlmostEqual(front_vel, 0.6,
                               msg="Wave velocity along the fibers direction is incorrect! (LR91 2D)",
                               delta=0.05)
    
    
    def test_wave_across_the_fibers(self):
        sys.stdout.write("---> Check the wave speed across the fibers\n")
        self.tissue.fibers[:,:,0] = 1.
        self.tissue.fibers[:,:,1] = 0.
        self.tissue.D_al  = 0.1
        self.tissue.D_ac  = 0.0111
    
        stim_params = [[0, 5, 0, 100, -20, 0.]]
        self.lr91.stim_params    = stim_params
    
        self.lr91.run()
    
        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
    
        self.assertAlmostEqual(front_vel, 0.2,
                               msg="Wave velocity across the fibers direction is incorrect! (LR91 2D)",
                               delta=0.05)

    # def test_fibrosis_velocity(self):
    #     sys.stdout.write("---> Check the fibrosis wave speed\n")
    #     self.lr91.Di = 0.1
    #     self.lr91.Dj = 0.1
    #     velocities = []
    #
    #     stim_params = [[0, 100, 0, 5, -20, 0.]]
    #     self.lr91.stim_params    = stim_params
    #
    #     self.lr91.run()
    #     velocities.append(np.mean(self.velocity_tracker.compute_velocity_front()))
    #
    #     self.tissue.add_pattern(Diffuse2DPattern(0, 100, 0, 100, 0.1))
    #     self.lr91.run()
    #     velocities.append(np.mean(self.velocity_tracker.compute_velocity_front()))
    #
    #     self.tissue.clean()
    #     self.tissue.add_pattern(Diffuse2DPattern(0, 100, 0, 100, 0.2))
    #     self.lr91.run()
    #     velocities.append(np.mean(self.velocity_tracker.compute_velocity_front()))
    #
    #     self.tissue.clean()
    #     self.tissue.add_pattern(Diffuse2DPattern(0, 100, 0, 100, 0.3))
    #     self.lr91.run()
    #     velocities.append(np.mean(self.velocity_tracker.compute_velocity_front()))
    #
    #     self.tissue.clean()
    #
    #     self.assertAlmostEqual(velocities[0], 0.7,
    #                            msg="Wave velocity in the presence of fibrosis is incorrect! (LR91 2D)",
    #                            delta=0.05)
    #     self.assertAlmostEqual(velocities[1], 0.65,
    #                            msg="Wave velocity in the presence of fibrosis is incorrect! (LR91 2D)",
    #                            delta=0.05)
    #     self.assertAlmostEqual(velocities[2], 0.62,
    #                            msg="Wave velocity in the presence of fibrosis is incorrect! (LR91 2D)",
    #                            delta=0.1)
    #     self.assertAlmostEqual(velocities[3], 0.52,
    #                            msg="Wave velocity in the presence of fibrosis is incorrect! (LR91 2D)",
    #                            delta=0.15)

    # def test_spiral_wave_period(self):
    #     sys.stdout.write("---> Check the spiral wave period\n")
    #     self.lr91.Di = 0.1
    #     self.lr91.Dj = 0.1
    #     self.lr91.t_max = 350
    #
    #     stim_params = [[0, 3, 0, 400, -20, 0.],
    #                    [0, 400, 0, 200, -20, 210.]]
    #     self.lr91.cardiac_tissue = self.tissue
    #     self.lr91.stim_params    = stim_params
    #
    #     period_tracker = Period2DTracker()
    #     period_tracker.target_model = self.lr91
    #     period_tracker.mode = "Detectors"
    #
    #     detectors = np.zeros([400, 400], dtype="uint8")
    #     positions = np.array([[100, 100]])
    #     detectors[positions[:, 0], positions[:, 1]] = 1
    #
    #     period_tracker.detectors = detectors
    #     period_tracker.threshold = -60
    #
    #     # add tracker to the model
    #     self.lr91.add_tracker(period_tracker)
    #
    #     self.lr91.run()
    #
    #     plt.imshow(self.lr91.u)
    #     plt.show()
    #
    #     print ("Period: ", period_tracker.output["100,100"][-1][1])
    #
    #     self.assertAlmostEqual(period_tracker.output["100,100"][-1][1], 100.,
    #                            msg="Spiral wave period is incorrect! (LR91 2D)",
    #                            delta=10.)

    # def test_apd(self):
    #     sys.stdout.write("---> Show the model apd\n")
    #     self.tissue.fibers[:,:,0] = 0.
    #     self.tissue.fibers[:,:,1] = 1.
    #
    #     stim_params = [[0, 200, 0, 3, -20, 0.]]
    #     self.lr91.stim_params    = stim_params
    #
    #     act_pot_tracker = ActionPotential2DTracker()
    #     act_pot_tracker.target_model = self.lr91
    #     act_pot_tracker.cell_ind = [30, 30]
    #     self.lr91.add_tracker(act_pot_tracker)
    #
    #     self.lr91.run()
    #
    #     plt.plot(np.arange(len(act_pot_tracker.act_pot))*self.lr91.dt, act_pot_tracker.act_pot )
    #     plt.show()
