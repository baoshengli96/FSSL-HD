from options import args_parser
import logging
import random
import numpy as np
from FedAvg import compute_delta, ServerUpdateNew
import torch
import torch.backends.cudnn as cudnn
import time
from local_supervised_resnet import SupervisedLocalUpdate
from local_unsupervised_resnet import UnsupervisedLocalUpdate
from networks.resnet import resnet18
from networks.models import FedAvgCNN
from tqdm import trange
from cifar_load import get_dataloader,  getdataset


def test(model,testdl):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testdl:
            _, images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)            
            correct += (predicted == labels.flatten()).sum().item() 
        acc = correct / total
    return acc  

if __name__ == '__main__':
    args = args_parser()

    supervised_user_id = [0]
    unsupervised_user_id = list(range(len(supervised_user_id), args.unsup_num + len(supervised_user_id)))
    sup_num = len(supervised_user_id)
    unsup_num = len(unsupervised_user_id)
    total_num = sup_num + unsup_num
    clt_ids = list(range(total_num))
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


    if args.dataset == 'SVHN':
        partition = torch.load('partition_strategy/SVHN_noniid_10%labeled.pth')
        net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar100':
        partition = torch.load('partition_strategy/cifar100_noniid_10%labeled.pth')
        net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar10':
        net_dataidx_map = torch.load('cifar10.pt')
    elif args.dataset == 'fmnist':
        net_dataidx_map = torch.load('fmnist.pt')        
      
    X_train, y_train, X_test, y_test = getdataset(args.dataset, args.datadir)        
        
    # X_train, y_train, X_test, y_test, _, traindata_cls_counts = partition_data_allnoniid(
    #     args.dataset, args.datadir, partition=args.partition, n_parties=total_num, beta=args.beta)


    # for client_idx in clt_ids:
    #     print('='*20)
    #     print("client %d sample num: %d" % (client_idx, len(net_dataidx_map[client_idx])))
    #     unique, counts = np.unique(y_train[net_dataidx_map[client_idx]], return_counts=True)
    #     label_counts = dict(zip(unique, counts))
    #     print("Each class num:")
    #     for label, count in label_counts.items():
    #         print(f"class {label}: {count} samples")  
        


    # print(X_train.shape)
    if args.dataset == 'SVHN':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])
        

    if args.dataset == 'cifar10' or args.dataset == 'SVHN' or args.dataset == 'fmnist':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    train_dl_locals = []
    for client_idx in clt_ids:
        if client_idx == 0:
            train_dl_local, _ = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                        y_train[net_dataidx_map[client_idx]],
                                                        args.dataset, args.datadir, args.batch_size,
                                                        is_labeled=True,
                                                        data_idxs=net_dataidx_map[client_idx],
                                                        pre_sz=args.pre_sz, input_sz=args.input_sz)    
        else:
            train_dl_local, _ = get_dataloader(args,
                                                    X_train[net_dataidx_map[client_idx]],
                                                    y_train[net_dataidx_map[client_idx]],
                                                    args.dataset,
                                                    args.datadir, args.batch_size, is_labeled=False,
                                                    data_idxs=net_dataidx_map[client_idx],
                                                    pre_sz=args.pre_sz, input_sz=args.input_sz)
        train_dl_locals.append(train_dl_local)
        
    if args.dataset == 'SVHN' or args.dataset == 'cifar100' or args.dataset == 'cifar10' or args.dataset == 'fmnist':
        test_dl, test_ds = get_dataloader(args, X_test, y_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
        
        
    # net_glob = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
    if args.dataset == 'SVHN' or args.dataset == 'fmnist':
        net_glob = FedAvgCNN()
        from local_supervised_resnet import SupervisedLocalUpdate
        from local_unsupervised_resnet import UnsupervisedLocalUpdate
    elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
        net_glob = resnet18(num_classes=n_classes)
        from local_supervised_cifar100 import SupervisedLocalUpdate
        from local_unsupervised_cifar100 import UnsupervisedLocalUpdate

    if args.resume:
        print('==> Resuming from checkpoint..')
        if args.dataset == 'cifar100':
            checkpoint = torch.load('warmup/cifar100.pth')
        elif args.dataset == 'SVHN':
            checkpoint = torch.load('warmup/SVHN.pth')

        net_glob.load_state_dict(checkpoint['state_dict'])
        start_epoch = 7
    else:
        start_epoch = 0

    # if len(args.gpu.split(',')) > 1:
    #     net_glob = torch.nn.DataParallel(net_glob, device_ids=[i for i in range(round(len(args.gpu) / 2))])  #
    net_glob.train()
    net_glob.cuda()
    # w_glob = copy.deepcopy(net_glob)
    v_glob = [torch.zeros_like(param).cuda() for param in net_glob.parameters()]
    w_locals = []
    # w_ema_unsup = []
    lab_trainer_locals = []
    unlab_trainer_locals = []
    sup_net_locals = []
    unsup_net_locals = []
    sup_optim_locals = []
    unsup_optim_locals = []

    total_lenth = sum([len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))])
    each_lenth = [len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))]
    client_freq = [len(net_dataidx_map[i]) / total_lenth for i in range(len(net_dataidx_map))]

    for i in supervised_user_id:
        lab_trainer_locals.append(SupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))

    for i in unsupervised_user_id:
        unlab_trainer_locals.append(
            UnsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))

    for com_round in trange(start_epoch, args.rounds):
        print("************* Comm round %d begins *************" % com_round)
        loss_locals = []
        delta_locals_this_round = []
        st = time.time()
        for client_idx in clt_ids:
            if client_idx in supervised_user_id:
                local = lab_trainer_locals[client_idx]
                train_dl_local = train_dl_locals[client_idx]
                w, loss = local.train(net_glob.state_dict(), train_dl_local)  
                delta_locals_this_round.append(compute_delta(w, net_glob.cuda()))
                loss_locals.append(loss)
                logging.info('Labeled client {}: training loss : {} '.format(client_idx, loss))

            else:
                local = unlab_trainer_locals[client_idx - sup_num]
                train_dl_local = train_dl_locals[client_idx]
                w, loss = local.train(net_glob.state_dict(), com_round * args.local_ep, client_idx, train_dl_local)                
                delta_locals_this_round.append(compute_delta(w, net_glob.cuda()))              
                loss_locals.append(loss)
                logging.info('UnLabeled client {}: training loss : {} '.format(client_idx, loss))

        net_glob, v_glob = ServerUpdateNew(net_glob.cuda(), delta_locals_this_round, client_freq, v_glob, args.beta1, loss_locals)


        loss_avg = sum(loss_locals) / len(loss_locals)
        logging.info( 
            '************ Loss Avg {}, LR {}, Round {} ends ************  '.format(loss_avg, args.base_lr, com_round))
        Accus_avg = test(net_glob, test_dl)
        logging.info("\nTEST: Epoch: {}".format(com_round))
        logging.info("\nConsumption time: {}".format(time.time()-st))
        
        logging.info(  "\nTEST Accus: {:6f}"
                    .format(Accus_avg))

