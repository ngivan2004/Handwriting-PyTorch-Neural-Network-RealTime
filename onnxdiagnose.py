import onnx

model = onnx.load('onnx_model.onnx')
print([input.name for input in model.graph.input])
