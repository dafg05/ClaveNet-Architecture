import torch

from cnarch.grooveTransformer import GrooveTransformer

def test_GrooveTransformer():
    model = GrooveTransformer(d_model = 8)
    src = torch.rand(4, 32, 27)

    h, v, o = model(src)

    assert h.shape == (4, 32, 9)
    assert v.shape == (4, 32, 9)
    assert o.shape == (4, 32, 9)

    print("Forward test passed.")

    h, v, o = model.inference(src)

    assert h.shape == (4, 32, 9)
    assert v.shape == (4, 32, 9)
    assert o.shape == (4, 32, 9)

    print("Inference test passed.")

if __name__ == "__main__":
    test_GrooveTransformer()
    

