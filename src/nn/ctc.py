from torch import nn
from torch.nn.modules import loss
from src.utils.metrice import WER, CER
from src.utils.text import TextUtility
from src.nn import TorchModule


class EncDecCTCModel(TorchModule):
    """CTC Encoder Decoder model
    """

    def __init__(self, encoder: TorchModule, decoder: TorchModule, blank_id: int, optimizer, utils: TextUtility):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss = nn.CTCLoss(blank=blank_id)
        self.optimizer = optimizer

        self.utils = utils
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
        probs = self.forward(x)
        probs = nn.functional.log_softmax(probs, dim=-1)
        loss = self.loss(probs, y, x_length, y_length)
    
        for y_probs, predicted_probs in zip(y, probs):
            y_probs = self.utils.convert_to_text(y_probs)
            prediction = self.utils.greedy(predicted_probs)
            prediction = self.utils.convert_to_text(prediction)

            self.cer(y_probs, prediction)
            self.wer(y_probs, prediction)

        self.log("CER", self.cer.average_errors)
        self.log("WER", self.wer.average_errors)
        self.log("loss", loss)
        