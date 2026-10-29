import unittest
import torch

from transformer.compositional import CompositionalConfig,CompositionalTransformer


class CompositionalTransformerTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(1)
        return CompositionalTransformer(CompositionalConfig(16,8,(2,3,3,4,3),2,2,16,0.)).eval()

    def test_sampling_logits_equal_teacher_forced_predictions(self):
        model=self.model(); conditions=torch.tensor([[1,2,0,3,1],[0,1,2,0,2]])
        seen=[]
        hook=model.lm_head.register_forward_hook(lambda module,args,output:seen.append(output.detach().clone()))
        tokens=model.generate(conditions,top_k=None)
        hook.remove()
        logits,loss=model(tokens,conditions)
        self.assertEqual(tuple(logits.shape),(2,8,16))
        self.assertTrue(torch.isfinite(loss))
        for position,actual in enumerate(seen):
            torch.testing.assert_close(actual,logits[:,position],rtol=1e-5,atol=1e-6)

    def test_future_image_tokens_do_not_leak(self):
        model=self.model(); conditions=torch.tensor([[1,2,0,3,1]])
        tokens=torch.arange(8).reshape(1,8)
        changed=tokens.clone(); changed[:,4:]=15
        left,_=model(tokens,conditions); right,_=model(changed,conditions)
        torch.testing.assert_close(left[:,:5],right[:,:5])

    def test_all_five_controls_receive_gradients(self):
        model=self.model(); tokens=torch.arange(8).reshape(1,8)
        conditions=torch.tensor([[1,2,0,3,1]])
        _,loss=model(tokens,conditions); loss.backward()
        for i,embedding in enumerate(model.condition_emb):
            gradient=embedding.weight.grad[conditions[0,i]]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(),0.)

    def test_checkpoint_configuration_round_trip(self):
        from dataclasses import asdict
        model=self.model()
        clone=CompositionalTransformer(CompositionalConfig(**asdict(model.cfg))).eval()
        clone.load_state_dict(model.state_dict())
        inputs=torch.arange(8).reshape(1,8); labels=torch.tensor([[1,2,0,3,1]])
        torch.testing.assert_close(model(inputs,labels)[0],clone(inputs,labels)[0])


if __name__=='__main__': unittest.main()
