import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, auc


def get_data_smiles(smiles_ids, smiles2vec):
    smiles_list = []
    for id in smiles_ids:
        smiles_vec = smiles2vec[int(id)]
        smiles_list.append(smiles_vec)
    smiles_list = np.array(smiles_list)
    return torch.LongTensor(smiles_list).cuda()


def get_data_seq(target_ids, word2vec):
    seq_list = []
    for id in target_ids:
        seq = word2vec[int(id)]
        seq_list.append(seq)
    seq_list = np.array(seq_list)
    return torch.LongTensor(seq_list).cuda()


# max_roc=0
def train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args, smiles2vec, seq2vec, drug_H_data, protein_H_data):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()
    loss_history = []
    max_auc = 0

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        data_s.to('cuda')
        data_a.to('cuda')

        
    # Train model
    lbl = data_a.y
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    for epoch in range(args.epochs):    # 30
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (label, inp) in enumerate(train_loader):
            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()
            
            drug_vec = get_data_smiles(inp[1], smiles2vec)
            target_vec = get_data_seq(inp[0], seq2vec)
            
            output, cla_os, cla_os_a, _ = model(data_o, data_s, data_a, inp, drug_vec, target_vec, drug_H_data, protein_H_data)
            
            y_pred_train.append(output)
            loss1 = F.nll_loss(torch.log(output), label)
            loss2 = b_xent(cla_os, lbl.float())
            loss3 = b_xent(cla_os_a, lbl.float())
            loss_train = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3

            loss_history.append(loss_train)
            loss_train.backward()
            optimizer.step()
            
            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
        
            '''log = torch.squeeze(m(output))
            loss1 = loss_fct(log, label.float())
            loss2 = b_xent(cla_os, lbl.float())
            loss3 = b_xent(cla_os_a, lbl.float())
            loss_train = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3

            loss_history.append(loss_train)
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()'''
            
            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))
        
        train_pred = torch.cat(y_pred_train)
        train_pred = train_pred.detach().cpu().numpy()
        roc_train = roc_auc_score(y_label_train, train_pred[:, 1])

        # roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation after each epoch
        if not args.fastmode:
            roc_val, prc_val, f1_val, loss_val = test(model, val_loader, data_o, data_s, data_a, args, smiles2vec, seq2vec, drug_H_data, protein_H_data)
            auroc_test, prc_test, f1_test, loss_test = test(model, test_loader, data_o, data_s, data_a, args, smiles2vec, seq2vec, drug_H_data, protein_H_data)
            if roc_val > max_auc:
                model_max = copy.deepcopy(model)
                # save model
                # torch.save(model_max.state_dict(), 'model/{}/best_model.pth'.format(args.in_file))
                max_auc = roc_val

            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(roc_train),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'auroc_val: {:.4f}'.format(roc_val),
                  'auprc_val: {:.4f}'.format(prc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'time: {:.4f}s'.format(time.time() - t))
            
            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_test: {:.4f}'.format(loss_test.item()),
                  'auroc_test: {:.4f}'.format(auroc_test),
                  'auprc_test: {:.4f}'.format(prc_test),
                  'f1_test: {:.4f}'.format(f1_test),
                  'time: {:.4f}s'.format(time.time() - t))
            
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    print(len(loss_history))
    loss_history_np = [tensor_item.cpu().detach().numpy() for tensor_item in loss_history]

    plt.plot(loss_history_np)
    plt.savefig('loss_curve.png') 
    # plt.show()
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test = test(model_max, test_loader, data_o, data_s, data_a, args, smiles2vec, seq2vec, drug_H_data, protein_H_data)
    
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))

    with open(args.out_file, 'a') as f:
        f.write('{0}\t{1}\t{6}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\n'.format(
            args.in_file, args.seed, loss_test.item(), auroc_test, prc_test, f1_test, args.feature_type))


def test(model, loader, data_o, data_s, data_a, args, smiles2vec, seq2vec, drug_H_data, protein_H_data):

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    lbl = data_a.y
    
    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            if args.cuda:
                label = label.cuda()
            
            drug_vec = get_data_smiles(inp[1], smiles2vec)
            target_vec = get_data_seq(inp[0], seq2vec)
            
            output, cla_os, cla_os_a, _ = model(data_o, data_s, data_a, inp, drug_vec, target_vec, drug_H_data, protein_H_data)
            
            y_pred.append(output)
            loss1 = F.nll_loss(torch.log(output), label)
            loss2 = b_xent(cla_os, lbl.float())
            loss3 = b_xent(cla_os_a, lbl.float())
            loss = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3
          
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            
            
        y_pred = torch.cat(y_pred)
        y_pred = y_pred.detach().cpu().numpy()
        auc = roc_auc_score(y_label, y_pred[:, 1])
        aupr = average_precision_score(y_label, y_pred[:, 1])
        f1 = f1_score(y_label, np.argmax(y_pred, axis=1))

    # roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss
    return auc, aupr, f1, loss
