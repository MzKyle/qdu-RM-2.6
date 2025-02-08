import torch
import torch.onnx
import os

print(os.getcwd())
 
def export(model, pthfile=r'./model.pth', input_shape=(1, 1, 20, 20), onnxfile=r'./model.onnx', local='cpu'):
    """
    将PyTorch模型导出为ONNX格式。
    
    参数:
        model: 要导出的PyTorch模型。
        pthfile: 预训练模型的路径，默认为当前目录下的'model.pth'。
        input_shape: 模型输入的形状，默认为(1, 1, 20, 20)。
        onnxfile: 导出的ONNX模型的路径，默认为当前目录下的'model.onnx'。
        local: 模型加载的设备，默认为'cpu'。
        
    """
    loaded_model = torch.load(pthfile, map_location=local)
    model.load_state_dict(loaded_model)
    input = torch.randn(input_shape)
    input_names = ["Armor"]
    output_names = ["Number"]
    torch.onnx.export(model, input, onnxfile, verbose=False, opset_version=12, input_names=input_names, output_names=output_names)
    