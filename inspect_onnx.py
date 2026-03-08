import onnx
import onnxruntime as onnxruntime
import sys

path = sys.argv[1]
model = onnx.load(path)


print("Input : \n")

for i in model.graph.input :
    dims = []

    for d in i.type.tensor_type.shape.dim:

        if d.dim_value != 0 :
            dims.append(d.dim_value)
        else :
            dims.append("?")

    print(" ", i.name, dims)


print("Output : \n")

for o in model.graph.output:
    dims = []

    for d in o.type.tensor_type.shape.dim:

        if d.dim_value != 0 :
            dims.append(d.dim_value)
        else :
            dims.append("?")

    print(" ", o.name, dims)




#sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
#print("\nRuntime outputs (names, shapes):")
#for o in sess.get_outputs():
#    print(" ", o.name, o.shape, o.type)
