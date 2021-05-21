import torch
import torch.nn as nn
import wandb
import numpy as np
import os
from tqdm import tqdm


def train(model, train_loader, val_loader, args):
    n_classes = {
        'mnist': [10],
        'fashion_mnist': [10],
        'celeba': [2]*train_loader.dataset.attr.shape[1]
    }[args.model.dataset_name]
    print(n_classes)
    hid_dim = {
        'mnist': args.model.z_dim*2,
        'fashion_mnist': args.model.z_dim*4,
        'celeba': args.model.z_dim*4,
    }[args.model.dataset_name]

    for clf in range(len(n_classes)):
        classifier = nn.Sequential(
            nn.Linear(args.model.z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, n_classes[clf]),
        )
        classifier = classifier.to('cuda:0')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,
                                                               factor=0.5, verbose=True)
        model = model.eval()
        model = model.to('cuda')
        best_loss = 1e10

        for epoch in tqdm(range(args.classifier.max_epoch)):
            logs = {
                'train_loss': 0,
                'val_loss': 0,
                'val_accuracy': 0,
                'epoch': epoch,
                'task': clf
            }

            ### Train
            running_loss = 0.
            classifier.train()
            for images, labels in train_loader:
                images = images.to('cuda:0')
                labels = labels.to('cuda:0')
                optimizer.zero_grad()
                z_mu, z_logvar = model.vae.q_z(images)
                z = model.vae.reparametrize(z_mu, z_logvar)
                output = classifier(z)
                if len(n_classes) > 1:
                    labels = labels[:, clf]
                loss = criterion(output, labels)
                loss.backward()

                optimizer.step()
                logs['train_loss'] += loss.item() / len(train_loader)

            ### Validation
            prop_correct = 0
            classifier.eval()
            with torch.no_grad():  # Turning off gradients to speed up
                for images, labels in val_loader:
                    images = images.to('cuda:0')
                    labels = labels.to('cuda:0')
                    z_mu, z_logvar = model.vae.q_z(images)
                    z = model.vae.reparametrize(z_mu, z_logvar)
                    output = classifier(z)
                    if len(n_classes) > 1:
                        labels = labels[:, clf]
                    logs['val_loss'] += criterion(output, labels).item() / len(val_loader)
                    prop_correct += sum(output.argmax(1) == labels)
            scheduler.step(logs['val_loss'])
            logs['val_accuracy'] = prop_correct / len(val_loader.dataset)
            wandb.log(logs)
            if logs['val_loss'] < best_loss:
                best_loss = logs['val_loss']
                if len(n_classes) > 1:
                    wandb.run.summary['best_val_loss_{}'.format(clf)] = best_loss
                    wandb.run.summary['best_val_acc_{}'.format(clf)] = logs['val_accuracy']
                    torch.save(classifier, os.path.join(wandb.run.dir, 'best_clf_{}.pth'.format(clf)))
                else:
                    wandb.run.summary['best_val_loss'] = best_loss
                    wandb.run.summary['best_val_acc'] = logs['val_accuracy']
                    torch.save(classifier, os.path.join(wandb.run.dir, 'best_clf.pth'))
