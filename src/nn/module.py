import torch
from pytorch_lightning import LightningModule

class TorchModule(LightningModule):

    def export_onnx(self, output_destination, input_shape):
        # export model in ONNX format
        torch.onnx.export(
            self, 
            torch.zeros(input_shape),
            output_destination
        )

    #TODO
    def download_model(self):
        pass