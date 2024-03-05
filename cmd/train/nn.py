from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import compute_class_weight
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric, F1Score, Precision, Recall, Accuracy

import simtools
from simtools.nn import ANN


class LitNNTrainer(pl.LightningModule):
    def __init__(self, d_in: int, d_out: int, pos_weight=None):
        super().__init__()
        self.save_hyperparameters()

        ann = ANN(
            d_input=d_in,
            d_output=d_out,
            d_hidden=384,
            n_hidden_layers=2,
            hidden_activation='gelu'
        )

        self.model = ann
        self.crit = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight))

        # Default parameters
        self.optimizer_type = 'lbfgs'
        self.use_scheduler = True
        self.lr = 5e-1
        self.optim_eps = 1e-12

        self.train_loss_metric = MeanMetric(nan_strategy='ignore')
        self.valid_loss_metric = MeanMetric(nan_strategy='ignore')
        self.test_loss_metric = MeanMetric(nan_strategy='ignore')
        self.train_f1_metric = F1Score(task='binary')
        self.valid_f1_metric = F1Score(task='binary')
        self.test_f1_metric = F1Score(task='binary')
        self.train_precision_metric = Precision(task='binary')
        self.valid_precision_metric = Precision(task='binary')
        self.test_precision_metric = Precision(task='binary')
        self.train_recall_metric = Recall(task='binary')
        self.valid_recall_metric = Recall(task='binary')
        self.test_recall_metric = Recall(task='binary')
        self.train_accuracy_metric = Accuracy(task='binary')
        self.valid_accuracy_metric = Accuracy(task='binary')
        self.test_accuracy_metric = Accuracy(task='binary')

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.parameters(), lr=self.lr, betas=(0.9, 0.999),
                eps=self.optim_eps)
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(), lr=self.lr, betas=(0.9, 0.999),
                eps=self.optim_eps, weight_decay=1e-5)
        elif self.optimizer_type == 'lbfgs':
            optimizer = optim.LBFGS(
                self.parameters(), lr=self.lr,
                tolerance_grad=self.optim_eps, tolerance_change=self.optim_eps * 1e-2,
                history_size=300, line_search_fn='strong_wolfe')
        else:
            raise TypeError(f'Optimization algorithm {self.optimizer_type} is not supported yet.')

        optimizer_config = {
            'optimizer': optimizer,
        }

        if self.use_scheduler:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', factor=0.2, patience=4, verbose=True)
            optimizer_config['lr_scheduler'] = lr_scheduler
            optimizer_config['monitor'] = 'val.loss'

        return optimizer_config

    def forward(self, inputs, targets=None, *args, **kwargs):
        output = self.model(inputs)
        if isinstance(output, torch.Tensor):
            output = {'logits': output}

        if targets is not None:
            loss = self.crit(output['logits'], targets)
            output['loss'] = loss
        return output

    def predict(self, inputs):
        # Note that y_scaler should be initialized before calling for predictions.
        output = self(inputs=inputs)
        if hasattr(self, 'y_scaler'):
            output['logits'].data = torch.tensor(
                self.y_scaler.inverse_transform(output['logits'].cpu()),
                dtype=inputs.dtype,
                device=inputs.device
            )
        return output

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        return self(inputs=inputs, targets=targets)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        return self(inputs=inputs, targets=targets)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        # Note that y_scaler should be set, before calling test methods.
        inputs, targets = batch
        output = self(inputs=inputs, targets=targets)
        if hasattr(self, 'y_scaler'):
            output['logits'].data = torch.tensor(
                self.y_scaler.inverse_transform(output['logits'].cpu()),
                dtype=inputs.dtype,
                device=inputs.device
            )
        return output

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        _, targets = batch
        loss = self.train_loss_metric(outputs['loss'])
        f1 = self.train_f1_metric(outputs['logits'], targets)
        prec = self.train_precision_metric(outputs['logits'], targets)
        rec = self.train_recall_metric(outputs['logits'], targets)
        acc = self.train_accuracy_metric(outputs['logits'], targets)
        self.log('train.loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train.f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train.prec', prec, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train.rec', rec, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train.acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        _, targets = batch
        loss = self.valid_loss_metric(outputs['loss'])
        f1 = self.valid_f1_metric(outputs['logits'], targets)
        prec = self.valid_precision_metric(outputs['logits'], targets)
        rec = self.valid_recall_metric(outputs['logits'], targets)
        acc = self.valid_accuracy_metric(outputs['logits'], targets)
        self.log('val.loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val.f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val.prec', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val.rec', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val.acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        _, targets = batch
        loss = self.test_loss_metric(outputs['loss'])
        f1 = self.test_f1_metric(outputs['logits'], targets)
        prec = self.test_precision_metric(outputs['logits'], targets)
        rec = self.test_recall_metric(outputs['logits'], targets)
        acc = self.test_accuracy_metric(outputs['logits'], targets)
        self.log('test.loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test.f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test.prec', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test.rec', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test.acc', acc, on_step=False, on_epoch=True, prog_bar=True)


def plot_preds(model, x_test, y_test, labels):
    output = model.predict(x_test)
    logits = output['logits']
    fig: plt.Figure
    ax: plt.Axes
    cmap1 = colormaps['Set3']
    fig, axes = plt.subplots(nrows=y_test.shape[-1])
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=np.object_)
    # fig.set_size_inches(10.8 * 2, 19.2 * 2)
    # fig.set_dpi(200)

    fmt = dict()  # markersize=8, linewidth=1
    for idx, ax in enumerate(axes):
        sl = slice(None, None, 50)
        xarr = np.arange(len(logits[sl, idx]))
        ax.plot(
            xarr,
            logits[sl, idx],
            '.-',
            label=f'PRED: {labels[idx]}',
            color=cmap1.colors[2 * idx],
            **fmt
        )
        ax.plot(
            xarr,
            y_test[sl, idx],
            '.-',
            label=f'REAL: {labels[idx]}',
            color=cmap1.colors[2 * idx + 1],
            **fmt
        )
        ax.legend()

    fig.tight_layout(w_pad=2, h_pad=2)
    plt.show()


def run_trainer(features=None):
    if features is None:
        features = slice(None)
    pl.seed_everything(42)

    # Params
    batch_size = 1024
    limit_train_batches = None
    limit_val_batches = None
    limit_test_batches = None

    root_dir = Path(
        '.cache', 'torch-lightning',
        LitNNTrainer.__name__.lower(),
    )
    df = pd.read_csv('data/features.v2.csv', sep=';')

    df = simtools.prep.misc.process_scrappy_dataset(df)

    meta = simtools.misc.read_meta()

    x_data = df[meta.all]
    y_data = df[meta.target]

    w = compute_class_weight(class_weight='balanced', classes=np.unique(y_data), y=y_data.to_numpy().ravel())
    w = {False: w[0], True: w[1]}

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    ct = ColumnTransformer(
        [
            (
                'cat',
                OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                meta.multicat,
            ),
            (
                'std',
                StandardScaler(),
                list(set(meta.all).difference(meta.multicat))
            ),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.set_output(transform='pandas')

    ct.fit(x_data)

    splits = list(cv.split(x_data, y_data))
    scores = {
        'f1': [],
        'prec': [],
        'rec': [],
        'acc': [],
    }

    for step, (train_indices, test_indices) in enumerate(splits):
        checkpoint_callback = ModelCheckpoint(
            filename='{epoch:04d}-{val.loss:.2f}-{val.f1:.4f}-{val.prec:.4f}-{val.rec:.4f}',
            monitor='val.loss', verbose=True, mode='min', save_top_k=8, save_on_train_epoch_end=True)
        early_stopping_callback = EarlyStopping(
            monitor='val.loss', min_delta=1e-5, mode='min', patience=16, verbose=True)

        callbacks = [
            checkpoint_callback,
            early_stopping_callback
        ]

        trainer = pl.Trainer(
            callbacks=callbacks,
            default_root_dir=str(root_dir),
            max_epochs=300,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            accelerator='cpu',
        )
        x_train = ct.transform(x_data.iloc[train_indices])
        y_train = y_data.iloc[train_indices]
        x_valid = ct.transform(x_data.iloc[test_indices])
        y_valid = y_data.iloc[test_indices]
        x_test = ct.transform(x_data.iloc[test_indices])
        y_test = y_data.iloc[test_indices]

        x_train = torch.tensor(x_train.loc[:, features].to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        x_valid = torch.tensor(x_valid.loc[:, features].to_numpy(), dtype=torch.float32)
        y_valid = torch.tensor(y_valid.to_numpy(), dtype=torch.float32)
        x_test = torch.tensor(x_test.loc[:, features].to_numpy(), dtype=torch.float32)
        y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

        train_dataloader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            TensorDataset(x_valid, y_valid),
            batch_size=batch_size,
            shuffle=False
        )
        test_dataloader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False
        )

        # Model training
        lit_model = LitNNTrainer(x_train.shape[-1], y_train.shape[-1], pos_weight=w[True])
        trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        if checkpoint_callback.best_model_path != '':
            print(f'Loading best model from: {checkpoint_callback.best_model_path}')
            lit_model = LitNNTrainer.load_from_checkpoint(checkpoint_callback.best_model_path)

        results = trainer.test(model=lit_model, dataloaders=test_dataloader)
        scores['f1'].append(results[-1]['test.f1'])
        scores['prec'].append(results[-1]['test.prec'])
        scores['rec'].append(results[-1]['test.rec'])
        scores['acc'].append(results[-1]['test.acc'])

    print(f'F1 score:         {np.asarray(scores["f1"]).mean()}')
    print(f'Precision score:  {np.asarray(scores["prec"]).mean()}')
    print(f'Recall score:     {np.asarray(scores["rec"]).mean()}')
    print(f'Accuracy score:   {np.asarray(scores["acc"]).mean()}')
    return scores


def main():
    meta = simtools.misc.read_meta()
    run_trainer()


if __name__ == '__main__':
    main()
