# install PyTorch
pip install torch==2.5.1+cu121 -f https://download.pytorch.org/whl/torch

# collect all test cases
python collect_tests.py --config-file `<path/to/config/file>`

# run collected tests
python run.py --config-file `<path/to/config/file>`