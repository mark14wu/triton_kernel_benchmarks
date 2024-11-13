pytest -s ../../liger_kernel/test/transformers/test_cross_entropy.py::test_correctness
pytest -s ../../liger_kernel/test/transformers/test_cross_entropy.py::test_correctness_functional
pytest -s ../../liger_kernel/test/transformers/test_cross_entropy.py::test_correctness_with_ignore_index
pytest -s ../../liger_kernel/test/transformers/test_cross_entropy.py::test_correctness_with_label_smoothing_once
pytest -s ../../liger_kernel/test/transformers/test_cross_entropy.py::test_correctness_with_label_smoothing_with_ignore_index_once
pytest -s ../../liger_kernel/test/transformers/test_cross_entropy.py::test_correctness_not_last_layer
pytest -s ../../liger_kernel/test/transformers/test_cross_entropy.py::test_large_no_exception