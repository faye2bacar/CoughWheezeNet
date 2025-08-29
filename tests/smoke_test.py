from src.model_crnn import CRNN
import torch

def test_forward():
    net = CRNN(n_mels=64, n_classes=3)
    x = torch.randn(2,1,300,64)
    logits, exa = net(x)
    assert logits.shape == (2,3)
    assert exa.shape[0] == 2
