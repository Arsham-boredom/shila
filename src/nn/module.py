import torch
from pytorch_lightning import LightningModule

class TorchModule(LightningModule):

    def save(self, path):
        torch.save(self.state_dict, path)

    def inference(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    #TODO
    def download_model(self):
        pass