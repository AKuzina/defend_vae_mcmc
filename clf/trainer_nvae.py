import torch
import torch.nn as nn
import wandb
import numpy as np
import os
from tqdm import tqdm


def train(model, train_loader, val_loader, task_ids, args, **kwargs):
    n_classes = [2]*len(task_ids)
    
    print(n_classes)
    
    # get input dim:
    x, y = iter(train_loader).__next__()
    z , _, _, _, _, _, _ = model(x[:1].to('cuda:0'), connect=args.model.connect,
                                           t=args.model.temp, return_samples=True)
    z = torch.stack(z, 1).reshape(1, -1) #1, connect, z_dim -> 1, total_dim
    inp_dim = z.shape[1]
    hid_dim = 512
    print(inp_dim, hid_dim)
    
    # create datasets of latent vectors
    train_z = []
    train_lab = []
    print('Preparing train dataset...')
    for images, labels in tqdm(train_loader):
        images = images.to('cuda:0')
        z = model.get_z(images, args.model.connect, args.model.temp)
        train_z.append(z.cpu())
        train_lab.append(torch.stack(labels, 1))
    train_dset = torch.utils.data.TensorDataset(torch.cat(train_z), torch.cat(train_lab))
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.nvae.batch_size)
    
    val_z = []
    val_lab = []
    print('Preparing val dataset...')
    for images, labels in tqdm(val_loader):
        images = images.to('cuda:0')
        z = model.get_z(images, args.model.connect, args.model.temp)
        val_z.append(z.cpu())
        val_lab.append(torch.stack(labels, 1))
    val_dset = torch.utils.data.TensorDataset(torch.cat(val_z), torch.cat(val_lab))
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.nvae.batch_size)
                                 

    for clf in range(len(n_classes)):
        TASK = task_ids[clf]
        classifier = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
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
        print(len(train_loader))
        for epoch in tqdm(range(args.classifier.max_epoch)):
            logs = {
                'train_loss': 0,
                'val_loss': 0,
                'val_accuracy': 0,
                'epoch': epoch,
                'task': clf
            }

            ### Train
            best_loss = 1e10
            running_loss = 0.
            classifier.train()
            for z, labels in train_loader:
                z = z.to('cuda:0')
                labels = labels[:, TASK].to('cuda:0')
                optimizer.zero_grad()
#                 z = model.get_z(images, args.model.connect, args.model.temp)
                output = classifier(z)
                loss = criterion(output, labels)
                loss.backward()

                optimizer.step()
                logs['train_loss'] += loss.item() / len(train_loader)

            ### Validation
            prop_correct = 0
            classifier.eval()
            with torch.no_grad():  # Turning off gradients to speed up
                for z, labels in val_loader:
                    z = z.to('cuda:0')
                    labels = labels[:, TASK].to('cuda:0')
#                     z = model.get_z(images, args.model.connect, args.model.temp)
                    output = classifier(z)
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
