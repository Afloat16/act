"""Normalization aggregates every real timestep of variable-length episodes."""
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from utils import get_norm_stats


class EpisodeNormStatsTest(unittest.TestCase):
    def write_episode(self, folder, index, qpos, actions):
        with h5py.File(Path(folder) / f"episode_{index}.hdf5", "w") as file:
            file["observations/qpos"] = qpos
            file["observations/qvel"] = np.zeros_like(qpos)
            file["action"] = actions

    def check_stats(self, qpos_episodes, action_episodes):
        with tempfile.TemporaryDirectory() as folder:
            for index, (qpos, actions) in enumerate(zip(qpos_episodes, action_episodes)):
                self.write_episode(folder, index, qpos, actions)
            result = get_norm_stats(folder, len(qpos_episodes))
        for prefix, episodes in (("qpos", qpos_episodes), ("action", action_episodes)):
            data = np.concatenate(episodes)
            ddof = 1 if len(data) > 1 else 0
            np.testing.assert_allclose(result[prefix + "_mean"], data.mean(axis=0))
            np.testing.assert_allclose(result[prefix + "_std"],
                                       np.maximum(data.std(axis=0, ddof=ddof), 1e-2),
                                       rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(result["example_qpos"], qpos_episodes[-1])

    def test_unequal_episode_lengths_use_timestep_weighted_statistics(self):
        qpos = [np.array([[1., 4.], [3., 6.]], dtype=np.float32),
                np.array([[9., 8.]], dtype=np.float32),
                np.array([[4., 3.], [8., 7.], [2., 9.]], dtype=np.float32)]
        actions = [episode * np.array([2., -0.5], dtype=np.float32) for episode in qpos]
        self.check_stats(qpos, actions)

    def test_equal_length_statistics_are_preserved(self):
        qpos = [np.array([[1., 2.], [3., 5.]], dtype=np.float32),
                np.array([[7., 11.], [13., 17.]], dtype=np.float32)]
        self.check_stats(qpos, [episode + 1 for episode in qpos])

    def test_single_timestep_has_finite_clipped_scale(self):
        self.check_stats([np.array([[5., 7.]], dtype=np.float32)],
                         [np.array([[2., 4.]], dtype=np.float32)])


if __name__ == "__main__":
    unittest.main()

