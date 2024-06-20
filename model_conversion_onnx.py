import torch
from model import NNPy


def main():
    model = NNPy()
    model.load_state_dict(torch.load('model_state_augmented.pth'))
    model.eval()
    dummy = torch.zeros(1, 28, 28)
    torch.onnx.export(model, dummy, "onnx_model_augmented.onnx", verbose=True)


if __name__ == "__main__":
    main()
