# import torchbenchmark.models.pytorch_struct
import torchbenchmark.models.pyhpc_turbulent_kinetic_energy
import torchbenchmark.models.pyhpc_equation_of_state
import torchbenchmark.models.pyhpc_isoneutral_mixing
import torchbenchmark.models.functorch_dp_cifar10
import torchbenchmark.models.opacus_cifar10
import torchbenchmark.models.functorch_maml_omniglot
import torchbenchmark.models.moco
import torchbenchmark.models.maml
import torchbenchmark.models.lennard_jones
import torchbenchmark.models.maml_omniglot
import torchbenchmark.models.basic_gnn_edgecnn
import torchbenchmark.models.basic_gnn_gcn
import torchbenchmark.models.basic_gnn_gin
import torchbenchmark.models.basic_gnn_sage

# def test_pytorch_struct():
#     model, example_inputs = torchbenchmark.models.pytorch_struct.Model(test="eval", device="cuda", batch_size=1).get_module()
#     model(*example_inputs)

def test_pyhpc_turbulent_kinetic_energy():
    model, example_inputs = torchbenchmark.models.pyhpc_turbulent_kinetic_energy.Model(test="eval", device="cuda").get_module()
    model(*example_inputs)

def test_pyhpc_equation_of_state():
    model, example_inputs = torchbenchmark.models.pyhpc_equation_of_state.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_pyhpc_isoneutral_mixing():
    model, example_inputs = torchbenchmark.models.pyhpc_isoneutral_mixing.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_functorch_dp_cifar10():
    model, example_inputs = torchbenchmark.models.functorch_dp_cifar10.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_opacus_cifar10():
    model, example_inputs = torchbenchmark.models.opacus_cifar10.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_functorch_maml_omniglot():
    model, example_inputs = torchbenchmark.models.functorch_maml_omniglot.Model(test="eval", device="cuda").get_module()
    model(*example_inputs)

def test_moco():
    model, example_inputs = torchbenchmark.models.moco.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_maml():
    model, example_inputs = torchbenchmark.models.maml.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_lennard_jones():
    model, example_inputs = torchbenchmark.models.lennard_jones.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_maml_omniglot():
    model, example_inputs = torchbenchmark.models.maml_omniglot.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_basic_gnn_edgecnn():
    model, example_inputs = torchbenchmark.models.basic_gnn_edgecnn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_basic_gnn_gcn():
    model, example_inputs = torchbenchmark.models.basic_gnn_gcn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_basic_gnn_gin():
    model, example_inputs = torchbenchmark.models.basic_gnn_gin.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_basic_gnn_sage():
    model, example_inputs = torchbenchmark.models.basic_gnn_sage.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)
