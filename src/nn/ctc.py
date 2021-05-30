from torch import nn
from torch.nn.modules import loss
from src.nn import TorchModule


class EncDecCTCModel(TorchModule):
    """CTC Encoder Decoder model
    """

    def __init__(self, encoder: TorchModule, decoder: TorchModule, blank_id: int, optimizer):
        super().__init__()

        self.model = nn.Sequential(encoder, decoder)
        self.loss = nn.CTCLoss(blank=blank_id)
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def training_step(self, train_batch, batch_idx):
        x, y, x_length, y_length = train_batch
        y_pred = self.model(x)
        loss = self.loss(x, y_pred, x_length, y_length)
        self.log(loss)

        return loss

    def validation_step(self, validation_batch, batch_idx):
        x, y, x_length, y_length = validation_batch
        y_pred = self.model(x)
        loss = self.loss(x, y_pred, x_length, y_length)
        self.logger(loss)

        