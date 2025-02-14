import torchbenchmark.models.drq
import torchbenchmark.models.soft_actor_critic
import torchbenchmark.models.LearningToPaint

def test_drq():
    model, example_inputs = torchbenchmark.models.drq.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_soft_actor_critic():
    model, example_inputs = torchbenchmark.models.soft_actor_critic.Model(test="eval", device="cuda").get_module()
    model(*example_inputs)

def test_LearningToPaint():
    model, example_inputs = torchbenchmark.models.LearningToPaint.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)
