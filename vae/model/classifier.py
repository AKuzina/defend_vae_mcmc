import wandb
import torch
import os
import copy
from tqdm import tqdm

def _train_classifier_fn(
        model, criterion, optimizer, dataloader, lr_scheduler, device, task_i
):
    best_acc = 0.0

    # Counter for epochs without improvement
    epochs_no_improvement = 0
    max_epochs_no_improvement = 20

    # iterate over epochs
    for epoch in range(500):
        print("Epoch {}/{}".format(epoch + 1, 500))
        print("-" * 30)

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0

            # iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)[:, task_i]

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    outputs = model(inputs)
                    preds = outputs.argmax(1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # log statistics of the epoch
            wandb.log(
                {f"accuracy_{task_i}_{phase}": epoch_acc,
                 f"clf_loss_{task_i}_{phase}": epoch_loss},
            )
            if phase == "validation":
                lr_scheduler.step(epoch_loss)
                if epoch_acc >= best_acc:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_acc = epoch_acc
                    wandb.run.summary[f'best_val_loss_{task_i}'] = epoch_loss
                    wandb.run.summary[f'best_val_acc_{task_i}'] = epoch_acc
                    torch.save(model, os.path.join(wandb.run.dir, f'best_clf_{task_i}.pth'))
                    # Reset counter of epochs without progress
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1
        if epochs_no_improvement == max_epochs_no_improvement:
            print(
                f"Stopping training due to {epochs_no_improvement} epochs of no improvement in validation accuracy."
            )
            break

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Return model and histories
    return best_model_wts
