import pprint
from pathlib import Path
import torch.optim
import torch.utils.data
from torch import nn
from datasets import *
from utils_fun import *
import torch
from torch.autograd import Variable
from torchvision import transforms
import json
from utils.logger import Logger
from models.ssa import SSA
from models.nic import NIC
from models.scacnn import SCACNN
import time
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence

# 定义配置参数
ckpt_dir = 'checkpoints/'
ckpt_path = 'checkpoints/scacnn-model-10.pkl'
gpu = '0'
start_epoch = 0
log_file = 'log.txt'
restore_train = False
fine_tune = False
num_epochs = 120
batch_size = 16
learning_rate = 0.001
num_workers = 1
data_folder = './data/caption data/'  # 存放数据文件的文件夹路径
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # 数据文件的基本名称
model = 'scacnn'
att_mode = 'cs'
embed_size = 100
hidden_size = 512
dropout = 0.5
print_freq = 100
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
best_bleu4 = 0.  # BLEU-4 score right now

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

log_dir = Path('./logs')
if not log_dir.is_dir():
    log_dir.mkdir()
log_path = log_dir / Path(log_file)

LogMaster = Logger(log_path)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def main():

    global batch_size,epochs_since_improvement,model,embed_size,hidden_size,att_mode,learning_rate,word_map,best_bleu4
    logger = LogMaster.get_logger('main')

    # Create checkpoint directory
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    word_map_file = 'data/caption data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    vocab_size = len(word_map)

    if restore_train:
        if not os.path.isfile(ckpt_path):
            print('checkpoint not found: ', ckpt_path)
            exit(-1)
        checkpoint = torch.load(ckpt_path)
        args_dict = checkpoint['args']
        batch_size = args_dict['batch_size']
        learning_rate = args_dict['learning_rate']
        att_mode = args_dict['att_mode']
        model = args_dict['model']
        embed_size = args_dict['embed_size']
        hidden_size = args_dict['hidden_size']
        cur_epoch = checkpoint['epoch']
        print('restore training from existing checkpoint')
        pprint.pprint(args_dict)
    else:
        cur_epoch = 0
        checkpoint = None

    logger.info('building data loader...')
    # Build data loader

    # 数据加载器
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    logger.info(f'building model {model}...')
    # Build the models

    if model == 'ssa':
        net = SSA(embed_dim=embed_size, lstm_dim=hidden_size, vocab_size=vocab_size,
                  dropout=dropout, fine_tune=fine_tune)
    elif model == 'nic':
        net = NIC(embed_dim=embed_size, lstm_dim=hidden_size, vocab_size=vocab_size,
                  dropout=dropout, fine_tune=fine_tune)
    elif model == 'scacnn':
        net = SCACNN(embed_dim=embed_size, lstm_dim=hidden_size, vocab_size=vocab_size,
                     dropout=dropout, att_mode=att_mode, fine_tune=fine_tune)
    else:
        net = None
        print('model name not found: ' + model)
        exit(-2)

    params = net.train_params
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    # Epochs
    for epoch in range(start_epoch, num_epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)
            if fine_tune:
                adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
                  net=net,
                  criterion=criterion,
                  optimizer=optimizer,
                  epoch=epoch,
                  logger=logger,
                  checkpoint=checkpoint)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                    net=net,
                                    criterion=criterion,
                                    logger=logger)
        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
            logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(model, ckpt_dir,logger, cur_epoch, att_mode, epoch, epochs_since_improvement, net, optimizer, recent_bleu4, is_best)


def train(train_loader, net, criterion, optimizer, epoch, logger, checkpoint):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param criterion: loss layer
    :param epoch: epoch number
    """

    net.train()
    net.zero_grad()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and model == 'scacnn':
            net = nn.DataParallel(net)
        net.cuda()


    if restore_train:
        print('restoring from checkpoint...')
        net.load_state_dict(checkpoint['net_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])

    logger.info('start training...')
    # Train the Models

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (images, captions, lengths) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # Forward prop.
        if fine_tune:
            images = to_var(images, requires_grad=True)
        else:
            images = to_var(images, requires_grad=False)
        captions = to_var(captions, requires_grad=False)
        net.zero_grad()
        # Forward, Backward and Optimize
        outputs, captions, lengths, sort_ind = net.forward(images, captions, lengths)
        targets = captions[:, 1:]
        # outputs = outputs.contiguous().view(-1, vocab_size)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)
        targets = pack_padded_sequence(targets, lengths, batch_first=True)
        outputs = outputs.data
        targets = targets.data
        loss = criterion(outputs, targets)
        mean_loss = loss.mean()
        losses.update(mean_loss.item(), sum(lengths))
        loss = torch.mean(loss)  # 计算所有损失的平均值
        loss.backward()
        optimizer.step()
        # Keep track of metrics
        top5 = accuracy(outputs, targets, 5)
        top5accs.update(top5, sum(lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
            #                                                               batch_time=batch_time,
            #                                                               data_time=data_time, loss=losses,
            #                                                               top5=top5accs))
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

def validate(val_loader, net, criterion,logger):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (images, captions, lengths, allcaps) in enumerate(val_loader):
            # Forward prop.
            if fine_tune:
                images = to_var(images, requires_grad=True)
            else:
                images = to_var(images, requires_grad=False)
            captions = to_var(captions, requires_grad=False)
            outputs, captions, lengths, sort_ind = net.forward(images, captions, lengths)
            targets = captions[:, 1:]
            scores_copy = outputs.clone()
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)
            targets = pack_padded_sequence(targets, lengths, batch_first=True)
            outputs = outputs.data
            targets = targets.data
            # Calculate loss
            loss = criterion(outputs, targets)
            # Keep track of metrics
            mean_loss = loss.mean()
            losses.update(mean_loss.item(), sum(lengths))
            top5 = accuracy(outputs, targets, 5)
            top5accs.update(top5, sum(lengths))
            batch_time.update(time.time() - start)
            start = time.time()
            if i % print_freq == 0:
                # print('Validation: [{0}/{1}]\t'
                #       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                #                                                                 loss=losses, top5=top5accs))
                logger.info('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        # print(
        #     '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
        #         loss=losses,
        #         top5=top5accs,
        #         bleu=bleu4))
        logger.info('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4

if __name__ == '__main__':
    main()
