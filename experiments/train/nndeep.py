from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric, F1Score, Precision, Recall, Accuracy

import simtools


class MyModel(nn.Module):
    def __init__(self, d_embed, d_feat, d_hidden_embed, d_hidden_feat, n_hidden_embed, n_hidden_feat, n_clf):
        super().__init__()

        embeds = [nn.Linear(d_embed, d_hidden_embed)] + [nn.Linear(d_hidden_embed, d_hidden_embed) for _ in range(n_hidden_embed)]
        self.embeds = nn.ModuleList(embeds)

        feats = [nn.Linear(d_feat, d_hidden_feat)] + [nn.Linear(d_hidden_feat, d_hidden_feat) for _ in range(n_hidden_feat)]
        self.feats = nn.ModuleList(feats)

        self.attn = nn.MultiheadAttention(embed_dim=d_hidden_embed + d_hidden_feat, num_heads=1)
        self.wmat = nn.Parameter(torch.ones(d_hidden_embed + d_hidden_feat))

        self.clf = nn.ModuleList([nn.Linear(d_hidden_embed + d_hidden_feat, d_hidden_embed + d_hidden_feat) for _ in range(n_clf)])

        self.head = nn.Linear(d_hidden_embed + d_hidden_feat, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, embed, feat):
        y_embeds = embed
        for layer in self.embeds:
            y_embeds = layer(y_embeds)
            y_embeds = F.relu(y_embeds)
            y_embeds = self.dropout(y_embeds)

        y_feats = feat
        for layer in self.feats:
            y_feats = layer(y_feats)
            y_feats = F.relu(y_feats)
            y_embeds = self.dropout(y_embeds)

        vec = torch.concat([y_embeds, y_feats], dim=-1)
        #attn, _ = self.attn(vec, vec, vec)
        #attn = vec * self.wmat

        for layer in self.clf:
            vec = layer(vec)
            vec = F.relu(vec)
            vec = self.dropout(vec)

        #vec = vec + attn

        logits = self.head(vec)
        return {
            'logits': logits
        }


class LitNNDeepClassifier(pl.LightningModule):
    def __init__(
            self,
            d_embed: int,
            d_feat: int,
            d_hidden_embed: int,
            d_hidden_feat: int,
            n_hidden_embed: int,
            n_hidden_feat: int,
            n_clf: int,
            pos_weight=None
    ):
        super().__init__()
        self.save_hyperparameters()

        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight)

        ann = MyModel(
            d_embed=d_embed,
            d_feat=d_feat,
            d_hidden_embed=d_hidden_embed,
            d_hidden_feat=d_hidden_feat,
            n_hidden_embed=n_hidden_embed,
            n_hidden_feat=n_hidden_feat,
            n_clf=n_clf
        )

        self.model = ann
        self.crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Default parameters
        self.optimizer_type = 'adamw'
        self.use_scheduler = True
        self.lr = 5e-5
        self.optim_eps = 1e-8

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

    def forward(self, embeds, feats, targets=None, *args, **kwargs):
        output = self.model(embeds, feats)
        if isinstance(output, torch.Tensor):
            output = {'logits': output}

        if targets is not None:
            loss = self.crit(output['logits'], targets)
            output['loss'] = loss
        return output

    def predict(self, embeds, feats):
        output = self(embeds=embeds, feats=feats)
        return output

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs_embeds, inputs_feats, targets = batch
        output = self(embeds=inputs_embeds, feats=inputs_feats, targets=targets)
        return output

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs_embeds, inputs_feats, targets = batch
        output = self(embeds=inputs_embeds, feats=inputs_feats, targets=targets)
        return output

    def test_step(self, batch, batch_idx, *args, **kwargs):
        inputs_embeds, inputs_feats, targets = batch
        output = self(embeds=inputs_embeds, feats=inputs_feats, targets=targets)
        return output

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        _, _, targets = batch
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
        _, _, targets = batch
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
        _, _, targets = batch
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


def run_trainer():
    pl.seed_everything(4242)

    # Params
    train_batch_size = 32
    val_batch_size = 1024
    test_batch_size = 1024
    limit_train_batches = None
    limit_val_batches = None
    limit_test_batches = None

    root_dir = Path(
        '.cache', 'torch-lightning',
        LitNNDeepClassifier.__name__.lower(),
    )
    df = pd.read_csv('data/features.v2.csv', sep=';')

    df = simtools.prep.misc.process_scrappy_dataset(df)

    meta = simtools.misc.read_meta()

    x_data = df[meta.best_v1]
    y_data = df[meta.target]

    w = compute_class_weight(class_weight='balanced', classes=np.unique(y_data), y=y_data.to_numpy().ravel())
    w = {False: w[0], True: w[1]}

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    ct = ColumnTransformer(
        [
            (
                'std',
                StandardScaler(),
                meta.best_v1
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
            monitor='val.loss', verbose=True, mode='min', save_top_k=16, save_on_train_epoch_end=True)
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

        x_train_embed = torch.tensor(x_train.loc[:, [feat for feat in meta.best_v1 if feat.startswith('vec_')]].to_numpy(), dtype=torch.float32)
        x_train_feat = torch.tensor(x_train.loc[:, [feat for feat in meta.best_v1 if not feat.startswith('vec_')]].to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        x_valid_embed = torch.tensor(x_valid.loc[:, [feat for feat in meta.best_v1 if feat.startswith('vec_')]].to_numpy(), dtype=torch.float32)
        x_valid_feat = torch.tensor(x_valid.loc[:, [feat for feat in meta.best_v1 if not feat.startswith('vec_')]].to_numpy(), dtype=torch.float32)
        y_valid = torch.tensor(y_valid.to_numpy(), dtype=torch.float32)
        x_test_embed = torch.tensor(x_test.loc[:, [feat for feat in meta.best_v1 if feat.startswith('vec_')]].to_numpy(), dtype=torch.float32)
        x_test_feat = torch.tensor(x_test.loc[:, [feat for feat in meta.best_v1 if not feat.startswith('vec_')]].to_numpy(), dtype=torch.float32)
        y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

        train_dataloader = DataLoader(
            TensorDataset(x_train_embed, x_train_feat, y_train),
            batch_size=train_batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            TensorDataset(x_valid_embed, x_valid_feat, y_valid),
            batch_size=val_batch_size,
            shuffle=False
        )
        test_dataloader = DataLoader(
            TensorDataset(x_test_embed, x_test_feat, y_test),
            batch_size=test_batch_size,
            shuffle=False
        )

        # Model training
        lit_model = LitNNDeepClassifier(
            x_train_embed.shape[-1],
            x_train_feat.shape[-1],
            d_hidden_embed=256,
            d_hidden_feat=128,
            n_hidden_embed=4,
            n_hidden_feat=2,
            n_clf=4,
            pos_weight=w[True],
        )
        trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        if checkpoint_callback.best_model_path != '':
            print(f'Loading best model from: {checkpoint_callback.best_model_path}')
            lit_model = LitNNDeepClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

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
    run_trainer()


if __name__ == '__main__':
    main()
