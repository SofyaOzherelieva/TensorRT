import onnx

model = onnx.load("mnist_cc.onnx")
print(onnx.helper.printable_graph(model.graph))