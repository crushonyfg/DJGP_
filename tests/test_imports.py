"""Import sanity check for the post-reorg src/ layout.

Requires the package to be installed via `pip install -e .` in the jumpGP
conda env (it pulls in torch and other heavy deps when modules are loaded).
"""
import importlib
import unittest


CORE_MODULES = [
    # src/shared/ — runner libs and shared utilities
    "shared.utils",
    "shared.utils1",
    "shared.deepgp",
    "shared.djgp_validation",
    "shared.maxmin_design",
    "shared.djgp_runner",
    "shared.jumpgp_runner",
    # src/djgp/ — main package
    "djgp",
    "djgp.variational",
    "djgp.minibatch",
    "djgp.active_learning",
    "djgp.acquisition_metrics",
    "djgp.jumpgp_bridge",
    "djgp.sir",
    # src/jumpgp/ — facade
    "jumpgp",
    "jumpgp.linear",
    "jumpgp.quadratic",
    # src/JumpGaussianProcess/ — vendored canonical
    "JumpGaussianProcess",
    "JumpGaussianProcess.JumpGP_LD",
    "JumpGaussianProcess.jumpgp",
    # src/data_gen/ — generators
    "data_gen.synthetic",
    "data_gen.highdata",
    "data_gen.highdata_utils",
    "data_gen.autoencoder",
    "data_gen.lh_autoencoder",
    "data_gen.uci_autoencoder",
    "data_gen.analysis",
]

EXPERIMENT_MODULES = [
    "experiments.uci.new_dataset",
    "experiments.erosion.erosion_exp_alone",
    "experiments.erosion.erosion_exp",
    "experiments.erosion.L2_erosion_exp_alone",
    "experiments.erosion.erosion_new_dataset",
    "experiments.synthetic.L2_jgp_alone_highdata",
    "experiments.synthetic.minibatch_LH",
    "experiments.synthetic.validation_new_dataset",
    "experiments.synthetic.experiment_new",
    "experiments.synthetic.compare_K",
    "experiments.synthetic.jgp_alone_highdata",
    "experiments.oht.OHTDataset_analysis",
    "experiments.oht.OHT_dataset_test",
    "experiments.uci.plot_uci_pca",
    "experiments.uci.plot_uci_pca_nonstationary",
    "experiments.active_learning.run_synth_al4jgp",
    "experiments.active_learning.run_synth_and_plot_acq",
    "experiments.active_learning.run_synth_and_plot_acq_v2",
]


class CorePackageImportTest(unittest.TestCase):
    def test_src_packages_import(self):
        for module_name in CORE_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


class ExperimentPackageImportTest(unittest.TestCase):
    def test_experiment_modules_import(self):
        for module_name in EXPERIMENT_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
