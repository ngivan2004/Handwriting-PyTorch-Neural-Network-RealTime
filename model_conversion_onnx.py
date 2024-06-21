import torch
from model import NNPy
from model import NNPyCNN


def main():
    model = NNPy()
    model.load_state_dict(torch.load('model_state.pth'))
    model.eval()
    dummy = torch.zeros(1, 28, 28)
    torch.onnx.export(model, dummy, "onnx_model.onnx", input_names=[
                      'input'], output_names=['output'], opset_version=9, verbose=True)

    model_cnn = NNPyCNN()
    model_cnn.load_state_dict(torch.load('model_state_augmented+cnn.pth'))
    model_cnn.eval()
    dummy_cnn = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(model_cnn, dummy_cnn, "onnx_model_augmented+cnn.onnx", input_names=[
                      'input'], output_names=['output'], opset_version=9, verbose=True)


if __name__ == "__main__":
    main()
