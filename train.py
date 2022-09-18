'''
Description     : Train model.
Paper           : Surgical-VQLA: Transformer with Gated Vision-Language Embedding for 
                  Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Lalithkumar Seenivasan, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
'''

import os
import argparse
import pandas as pd
from lib2to3.pytree import convert

from torch import nn
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
from torch.utils.data  import DataLoader

from utils import *
from dataloader import *
from models.VisualBertPrediction import VisualBertPrediction
from models.VisualBertResMLPPrediction import VisualBertResMLPPrediction
from models.LViTPrediction import LViTPrediction

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


'''
Seed randoms
'''
def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    
    model.train()
    
    total_loss = 0.0    
    total_loss_class = 0.0
    total_loss_bbox = 0.0    
    label_true = None
    label_pred = None
    label_score = None
    bbox_outputs_pred = None
    bbox_label_true = None    
        
    for i, (_, visual_features, q, labels, bbox_label) in enumerate(train_dataloader,0):

        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=args.question_len)

    
        # GPU / CPU
        visual_features = visual_features.to(device)
        labels = labels.to(device)
        bbox_label = bbox_label.to(device)
                
        (classification_outputs, bbox_outputs) = model(inputs, visual_features)
        loss_class = criterion(classification_outputs, labels)
        loss_bbox = loss_giou_l1(bbox_outputs, bbox_label)  
        loss = loss_class + loss_bbox        
        
        # zero the parameter gradients
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        total_loss_class += loss_class.item()
        total_loss_bbox += loss_bbox.item()
        
        scores, predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
        label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
        label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)
        bbox_outputs_pred = bbox_outputs.data.cpu() if bbox_outputs_pred == None else torch.cat((bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        bbox_label_true = bbox_label.data.cpu() if bbox_label_true == None else torch.cat((bbox_label_true, bbox_label.data.cpu()), 0)
        
    # loss and acc
    acc, c_acc = calc_acc(label_true, label_pred), calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(bbox_label_true), box_cxcywh_to_xyxy(bbox_outputs_pred))
    
    print('Train: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f' %(epoch, total_loss, acc, precision, recall, fscore, bbox_miou))
    return acc


def validate(args, val_loader, model, criterion, epoch, tokenizer, device, save_output = False):
    
    model.eval()

    total_loss = 0.0    
    total_loss_class = 0.0
    total_loss_bbox = 0.0  
    label_true = None
    label_pred = None
    label_score = None
    bbox_outputs_pred = None
    bbox_label_true = None
    file_names = list()
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (file_name, visual_features, q, labels, bbox_label) in enumerate(val_loader,0):
            # prepare questions
            questions = []
            for question in q: questions.append(question)
            inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=args.question_len)

            # GPU / CPU
            visual_features = visual_features.to(device)
            labels = labels.to(device)
            bbox_label = bbox_label.to(device)
                                
            (classification_outputs, bbox_outputs) = model(inputs, visual_features)
            loss_class = criterion(classification_outputs,labels)
            loss_bbox = loss_giou_l1(bbox_outputs, bbox_label)  
            loss = loss_class + loss_bbox
                        
            total_loss += loss.item()
            total_loss_class += loss_class.item()
            total_loss_bbox += loss_bbox.item()  
                    
            scores, predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
            label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
            label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
            label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)
            bbox_outputs_pred = bbox_outputs.data.cpu() if bbox_outputs_pred == None else torch.cat((bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
            bbox_label_true = bbox_label.data.cpu() if bbox_label_true == None else torch.cat((bbox_label_true, bbox_label.data.cpu()), 0)
            
            for f in file_name: file_names.append(f)
            
    acc = calc_acc(label_true, label_pred) 
    c_acc = 0.0
    # c_acc = calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(bbox_label_true), box_cxcywh_to_xyxy(bbox_outputs_pred))    
    
    print('Test: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f| mIoU: %.6f' %(epoch, total_loss, acc, precision, recall, fscore, bbox_miou))

    if save_output:
        '''
            Saving predictions
        '''
        bbox_outputs_pred_org = box_cxcywh_to_xyxy(bbox_outputs_pred * torch.tensor([1280, 1024, 1280, 1024])) + torch.tensor([1, 1, 1, 1])
        bbox_label_true_org = box_cxcywh_to_xyxy(bbox_label_true * torch.tensor([1280, 1024, 1280, 1024])) + torch.tensor([1, 1, 1, 1])
        if os.path.exists(args.checkpoint_dir + 'text_files') == False:
            os.mkdir(args.checkpoint_dir + 'text_files' ) 
        file1 = open(args.checkpoint_dir + 'text_files/labels.txt', 'w')
        file1.write(str(label_true))
        file1.close()

        file1 = open(args.checkpoint_dir + 'text_files/predictions.txt', 'w')
        file1.write(str(label_pred))
        file1.close()

        if args.dataset_type == 'm18':
            convert_arr = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                            'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                            'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                            'left-top', 'right-top', 'left-bottom', 'right-bottom']

        df = pd.DataFrame(columns=["Img", "Ground Truth", "Prediction", "Bbox Ground Truth", "Bbox Prediction", "mIoU"])
        single_bbox_miou=[None] * len(label_true)
        for i in range(len(label_true)):
            single_bbox_miou[i] = mIoU_single(box_cxcywh_to_xyxy(bbox_label_true[i]), box_cxcywh_to_xyxy(bbox_outputs_pred[i]))
            df = df.append({'Img': file_names[i], 'Ground Truth': convert_arr[label_true[i]], 'Prediction': convert_arr[label_pred[i]],
                           'Bbox Ground Truth': bbox_label_true_org[i], 'Bbox Prediction': bbox_outputs_pred_org[i],
                           'mIoU': single_bbox_miou[i]}, ignore_index=True)
        
        df.to_csv(args.checkpoint_dir + args.checkpoint_dir.split('/')[1] + '_' + args.checkpoint_dir.split('/')[2] + '_eval.csv')
    
    return (acc, c_acc, precision, recall, fscore, bbox_miou)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VisualQuestionLocalizedAnswering')
    
    # Model parameters
    parser.add_argument('--emb_dim',        type=int,   default=300,                                help='dimension of word embeddings.')
    parser.add_argument('--n_heads',        type=int,   default=8,                                  help='Multi-head attention.')
    parser.add_argument('--dropout',        type=float, default=0.1,                                help='dropout')
    parser.add_argument('--encoder_layers', type=int,   default=6,                                  help='the number of layers of encoder in Transformer.')
    
    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=80,                                 help='number of epochs to train for (if early stopping is not triggered).') #80, 26
    parser.add_argument('--batch_size',     type=int,   default=64,                                 help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,                                  help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--print_freq',     type=int,   default=100,                                help='print training/validation stats every __ batches.')
    
    # existing checkpoint
    parser.add_argument('--checkpoint',     default=None,                                           help='path to checkpoint, None if none.')
    
    parser.add_argument('--lr',             type=float, default=0.00001,                            help='0.000005, 0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default= 'checkpoints/lvit/',                           help='/vb/vbrm/lvit/') 
    parser.add_argument('--dataset_type',   default= 'endovis',                                     help='endovis')
    parser.add_argument('--transformer_ver',default= 'lvit',                                        help='vb/vbrm/lvit')
    parser.add_argument('--patch_size',     default= 5,                                             help='1/2/3/4/5')
    parser.add_argument('--question_len',   default= 25,                                            help='25')
    parser.add_argument('--num_class',      default= 2,                                             help='25')
    parser.add_argument('--validate',       default=False,                                          help='When only validation required False/True')
    args = parser.parse_args()

    # load checkpoint, these parameters can't be modified
    final_args = {"emb_dim": args.emb_dim, "n_heads": args.n_heads, "dropout": args.dropout, "encoder_layers": args.encoder_layers}
    
    seed_everything()
    
    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print('device =', device)

    # best model initialize
    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0
    
    best_accuracy = 0
    best_accuracy_epoch = 0
    best_miou = 0
    best_miou_epoch = 0
    best_avg_acc_miou = 0
    best_avg_acc_miou_epoch = 0
    
    # dataset
    if args.dataset_type == 'endovis':
        '''
        Train and test dataloader for EndoVis18 & EndoVis17
        '''
        # tokenizer
        tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-EndoVis-18-VQA/')
        
        # data location
        
        # data location
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]
        
        folder_head = 'dataset/EndoVis-18-VQA/seq_'
        folder_tail = '/vqla/label/*.txt'
          
        # dataloader
        train_dataset = EndoVis18VQAClassification(train_seq, folder_head, folder_tail, patch_size = args.patch_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)
        val_dataset = EndoVis18VQAClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)
        
        # external dataloader
        external_folder_head = 'dataset/EndoVis-17-VQA/'
        external_folder_tail = '/vqla/label/*.txt'        
        external_val_dataset = EndoVis17VQAClassificationValidation(external_folder_head, external_folder_tail, patch_size = args.patch_size)
        external_val_dataloader = DataLoader(dataset=external_val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 18

    # Initialize / load checkpoint
    if args.checkpoint is None:
        # model
        if args.transformer_ver == 'vb':
            model = VisualBertPrediction(vocab_size=len(tokenizer), layers=args.encoder_layers, n_heads=args.n_heads, num_class = args.num_class)
        elif args.transformer_ver == 'vbrm':
            model = VisualBertResMLPPrediction(vocab_size=len(tokenizer), layers=args.encoder_layers, n_heads=args.n_heads, num_class = args.num_class, token_size = int(args.question_len+(args.patch_size * args.patch_size)))
        elif args.transformer_ver == 'lvit':
            model = LViTPrediction(vocab_size=len(tokenizer), layers=args.encoder_layers, n_heads=args.n_heads, num_class = args.num_class)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_Acc = checkpoint['Acc']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        final_args = checkpoint['final_args']
        for key in final_args.keys(): args.__setattr__(key, final_args[key])

    # Move to GPU, if available
    model = model.to(device)
    print(final_args)    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    # Loss function for classification head
    criterion = nn.CrossEntropyLoss().to(device)

    # validation
    if args.validate:
        checkpoint_path = args.checkpoint_dir + 'Best.pth'
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        test_acc, test_c_acc, test_precision, test_recall, test_fscore, test_bbox_miou = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=(args.epochs-1), tokenizer = tokenizer, device = device)
        ext_test_acc, ext_test_c_acc, ext_test_precision, ext_test_recall, ext_test_fscore, ext_test_bbox_miou = validate(args, val_loader=external_val_dataloader, model = model, criterion=criterion, epoch=(args.epochs-1), tokenizer = tokenizer, device = device)
    else:     
        for epoch in range(start_epoch, args.epochs):

            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)
            
            # train
            train_acc = train(args, train_dataloader=train_dataloader, model = model, criterion=criterion, optimizer=optimizer, epoch=epoch, tokenizer = tokenizer, device = device)

            # validation
            test_acc, test_c_acc, test_precision, test_recall, test_fscore, test_bbox_miou = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device)
            test_avg_acc_miou = (test_acc + test_bbox_miou) / 2                        
            
            if test_acc >= best_results[0]:
                epochs_since_improvement = 0
                
                best_results[0] = test_acc
                best_epoch[0] = epoch
                best_acc_epoch_fscore = test_fscore
                best_acc_epoch_miou = test_bbox_miou
                best_acc_epoch_avg_acc_miou = test_avg_acc_miou
                        
            else:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            
            print('Best Acc epoch:   %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_epoch[0], best_results[0], best_acc_epoch_fscore, best_acc_epoch_miou, best_acc_epoch_avg_acc_miou))

            if test_bbox_miou >= best_miou:
                best_miou = test_bbox_miou
                best_miou_epoch = epoch
                best_miou_epoch_acc = test_acc
                best_miou_epoch_fscore = test_fscore
                best_miou_epoch_avg_acc_miou = test_avg_acc_miou
            
            print('Best mIoU epoch:  %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_miou_epoch, best_miou_epoch_acc, best_miou_epoch_fscore, best_miou, best_miou_epoch_avg_acc_miou))         

            if test_avg_acc_miou >= best_avg_acc_miou:
                best_avg_acc_miou = test_avg_acc_miou
                best_avg_acc_miou_epoch = epoch
                best_avg_epoch_acc = test_acc
                best_avg_epoch_fscore = test_fscore
                best_avg_epoch_miou = test_bbox_miou

                checkpoint_path = args.checkpoint_dir + 'Best.pth'
                torch.save(model.state_dict(), checkpoint_path)
            
            print('Best Avg epoch:   %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_avg_acc_miou_epoch, best_avg_epoch_acc, best_avg_epoch_fscore, best_avg_epoch_miou, best_avg_acc_miou))
