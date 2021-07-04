from torch import nn
from torch.nn.modules import loss
from src.utils.metrice import WER, CER
from src.nn import TorchModule


class EncDecCTCModel(TorchModule):
    """CTC Encoder Decoder model
    """

    def __init__(self, encoder: TorchModule, decoder: TorchModule, blank_id: int, optimizer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss = nn.CTCLoss(blank=blank_id)
        self.optimizer = optimizer

        self.wer = WER()
        self.cer = CER()

    def forward(self, x):
        encoder_output = self.encoder(x)\
            .transpose(2, 1)

        probs = self.decoder(encoder_output)\
            .transpose(0, 1)

        return probs

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def training_step(self, train_batch, batch_idx):
        x, y, x_length, y_length = train_batch

        # why ?
        x_length = [i//2 for i in x_length]

        encoder_output = self.encoder(x)\
            .transpose(2, 1) #(batch, sequence, features)

        probs = self.decoder(encoder_output)\
            .transpose(0, 1) #(sequence, batch, features)

        probs = nn.functional.log_softmax(probs, dim=-1)

        loss = self.loss(probs, y, x_length, y_length)
        self.log("loss", loss)

        return loss

    def validation_step(self, validation_batch, batch_idx):
        x, y, x_length, y_length = validation_batch
        y_pred = self.model(x)
        loss = self.loss(x, y_pred, x_length, y_length)
        self.cer(y, y_pred)
        self.wer(y, y_pred)

        self.log("loss", loss)
        self.log("CER", self.cer.average_errors)
        self.log("WER", self.wer.average_errors)
        