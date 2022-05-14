import torch
from .general import message, set_random_seed, current_time
from copy import deepcopy
from .data import save_checkpoint
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

def forward_pass_dataloader(model, dataloader, optimizer, criterion, scheduler, device, random_seed, phase):
    #https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    set_random_seed(random_seed)
    running_loss = 0.0
    # running_correct = 0.0
    # total = 0
    total_labels = []
    total_preds = []

    if phase == 'train':
        model.train()
        for i, (images, labels, uid) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                for s in scheduler:
                    if s.__class__.__name__ in ['OneCycleLR', 'CyclicLR']:
                        s.step()
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item()
            # running_correct += torch.sum(preds == labels.data)
            total_labels += labels.cpu().numpy().tolist()
            total_preds += preds.cpu().numpy().tolist()
            # total += len(labels.data)

    elif phase == 'valid':
        model.eval()
        with torch.no_grad():
            for i, (images, labels, uid) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.float())
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                running_loss += loss.item()
                # running_correct += torch.sum(preds == labels.data)
                total_labels += labels.cpu().numpy().tolist()
                total_preds += preds.cpu().numpy().tolist()
                # total += len(labels.data)

    loss = running_loss / len(dataloader)
    # accuracy = (running_correct.double() / len(dataloader.dataset)).item()
    # return loss, accuracy, running_correct, total, total_labels, total_preds
    return loss, total_labels, total_preds


def fit_neural_network(classifier, train_metric='accuracy', valid_metric='accuracy', epochs=20, valid=True, print_every= 1, target_valid_metric='lowest', auto_save_ckpt=False, verbose=0):

    if hasattr (classifier, 'current_epoch'):
        start_epoch=classifier.current_epoch+1
        message(" Resuming training starting at epoch "+ str(start_epoch)+" on "+str(classifier.device))
        classifier.optimizer.load_state_dict(classifier.checkpoint['optimizer_state_dict'])
        message(" Optimizer state loaded successfully.")
    else:
        message(" Starting model training on "+str(classifier.device))
        start_epoch=0
        classifier.train_losses, classifier.valid_losses = [], []
        classifier.train_metric, classifier.valid_metric = [], []
        classifier.valid_metric_min = np.Inf

        if target_valid_metric == 'lowest':
            classifier.target_valid_metric = np.Inf
        else:
            classifier.target_valid_metric = target_valid_metric

        message(" Target Validation Metric set to "+ str(classifier.valid_metric_min))


    classifier.original_model = deepcopy(classifier.model)
    classifier.model = classifier.model.to(classifier.device)

    for e in tqdm(range(start_epoch,epochs), desc='Traninig Model: '):
        epoch_train_loss, epoch_train_labels, epoch_train_preds  = forward_pass_dataloader(classifier.model, classifier.dataloader_train, classifier.optimizer, classifier.criterion, classifier.scheduler,classifier.device, classifier.random_seed, phase='train')
        if train_metric == 'loss':
            epoch_train_metric = epoch_train_loss
        else:
            epoch_train_metric = calculate_metric(metric=train_metric, pred=epoch_train_preds, target=epoch_train_labels)
        classifier.train_losses.append(epoch_train_loss)
        classifier.train_metric.append(epoch_train_metric)

        if valid:
            epoch_valid_loss, epoch_valid_labels, epoch_valid_preds = forward_pass_dataloader(classifier.model, classifier.dataloader_valid, classifier.optimizer, classifier.criterion, classifier.scheduler, classifier.device, classifier.random_seed, phase='valid')
            if valid_metric == 'loss':
                epoch_valid_metric = epoch_valid_loss
            else:
                epoch_valid_metric = calculate_metric(metric=valid_metric, pred=epoch_valid_preds, target=epoch_valid_labels)
            classifier.valid_losses.append(epoch_valid_loss)
            classifier.valid_metric.append(epoch_valid_metric)


            if epoch_valid_metric < classifier.valid_metric_min:
                classifier.valid_metric_min = epoch_valid_metric
                classifier.best_model = deepcopy(classifier.model)
                if epoch_valid_metric <= classifier.target_valid_metric:
                    if auto_save_ckpt:
                        save_checkpoint(classifier=classifier, epochs=epochs, current_epoch=e)
                        save_ckpt, v_metric_dec, v_metric_below_target = True, True, True
                    else:
                        save_ckpt, v_metric_dec, v_metric_below_target = False, True, True
                else:
                    save_ckpt, v_metric_dec, v_metric_below_target = False, True, False
            else:
                save_ckpt, v_metric_dec, v_metric_below_target = False, False, False

            if e % print_every == 0:
                if verbose == 0:
                    message (
                            " epoch: {:4}/{:4} |".format(e, epochs)+
                            " t_loss: {:.5f} |".format(epoch_train_loss)+
                            " v_loss: {:.5f} (best: {:.5f}) |".format(epoch_valid_loss,np.min(classifier.valid_losses))
                    )

                if verbose == 1:
                    message (
                            " epoch: {:4}/{:4} |".format(e, epochs)+
                            " t_loss: {:.5f} |".format(epoch_train_loss)+
                            " v_loss: {:.5f} (best: {:.5f}) |".format(epoch_valid_loss,np.min(classifier.valid_losses))+
                            " t_metric ({}): {:.5f} |".format(train_metric, epoch_train_metric)+
                            " v_metric ({}): {:.5f} (best: {:.5f})|".format(valid_metric, epoch_valid_metric, classifier.valid_metric_min)

                    )

        else:
            if e % print_every == 0:
                if verbose != 0:
                    message(
                            " epoch: {:4}/{:4} |".format(e, epochs)+
                            " t_loss: {:.5f} |".format(epoch_train_loss)
                            )

        metrics_dict = {'train_loss':epoch_train_loss, 'train_metric':epoch_train_metric, 'valid_loss':epoch_valid_loss, 'valid_metric':epoch_valid_metric}

        if classifier.scheduler != [None]:
            for s in classifier.scheduler:
                if s.__class__.__name__ in ['OneCycleLR', 'CyclicLR']:
                    pass
                elif s.__class__.__name__ =='ReduceLROnPlateau':
                    s.step(metrics_dict[classifier.scheduler_metric])
                else:
                    s.step()

        if e+1 == epochs:
            message(' Training Finished Successfully!')

    if classifier.valid_metric_min > classifier.target_valid_metric:
        message(current_time()+" CAUTION: Achieved minimum validation metric "+str(classifier.valid_metric_min), " is not less than the set target metric of "+str(classifier.target_valid_metric))

    classifier.train_logs=pd.DataFrame({"train_loss": classifier.train_losses,
                                        "valid_loss" : classifier.valid_losses,
                                        "train_metric" : classifier.train_metric,
                                        "valid_metric" : classifier.valid_metric
                                        })


def pass_image_via_nn(tensor, model, device, output='logits', top_k=1):
    '''
    Runs image/images through model. The expected image(s) tensor shape is (B, C, W, H). If only 1 image to be passed, then B=1.
    output:
            logits: return logits by last layer of model per each image
            softmax: returns logits passed via softmax layer per each image
            topk: return list of predicted index/label and prediction percent per each image as per top_k specified
    '''
    model = model.to(device)
    tensor = tensor.to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor).cpu().detach()
        if output == 'logits':
            return out
        else:
            m = nn.Softmax(dim=1)
            predictions = m(out)
            if output == 'softmax':
                return predictions
            elif output == 'topk':
                out = []
                for i in predictions:
                    pred = torch.topk(i,k=top_k)
                    out.append([pred.indices.numpy().tolist(), pred.values.numpy().tolist()])
                return out


def pass_loader_via_nn(loader, model, device, output='logits', top_k=1, table=False):
    '''
    Same as pass_image_via_nn but for whole loader.
    table: in case of top_k =1, user can export a table with true labels, pred, perc and uid for each instance in loader.
    '''
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        label_list = []
        uid_list = []
        pred_list = []
        perc_list = []
        for i, (imgs, labels, uid) in enumerate(loader):
            label_list = label_list+labels.tolist()
            uid_list = uid_list+uid.tolist()
            imgs = imgs.to(device)
            out = model(imgs.float()).cpu().detach()
            if output == 'logits':
                return out
            else:
                m = nn.Softmax(dim=1)
                predictions = m(out)
                if output == 'softmax':
                    return predictions

                elif output == 'topk':
                    out =  []
                    for i in predictions:
                        pred = torch.topk(i,k=top_k)
                        out.append([pred.indices.numpy().tolist(), pred.values.numpy().tolist()])
                        if top_k == 1:
                            pred_list.append(pred.indices.item())
                            perc_list.append(pred.values.item())
                    if table==True:
                        return pd.DataFrame(list(zip(uid_list, label_list,pred_list, perc_list)),columns =['uid','label_id', 'pred_id', 'percent'])
                    else:
                        return out


def model_details(model, list=False, batch_size=1, channels=3, img_dim=224):
    if isinstance(model, str): model = eval('models.'+model+ "()")
    if list:
        return list(model.named_children())
    else:
        return summary(model, input_size=(batch_size, channels, img_dim, img_dim), depth=channels, col_names=["input_size", "output_size", "num_params"],)


def calculate_metric(metric, pred, target):
    if isinstance(metric, str):
        metrics_dict =  {
                'accuracy': accuracy_score(y_true=target, y_pred=pred),
                'micro_precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
                'micro_recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
                'micro_f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
                'macro_precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
                'macro_recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
                'macro_f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
                'balanced_accuracy_score': balanced_accuracy_score(y_true=target, y_pred=pred)
                # 'samples_precision': precision_score(y_true=target, y_pred=pred, average='samples'),
                # 'samples_recall': recall_score(y_true=target, y_pred=pred, average='samples'),
                # 'samples_f1': f1_score(y_true=target, y_pred=pred, average='samples'),
                }
        return metrics_dict[metric]
    else:
        return metric(pred, target)
