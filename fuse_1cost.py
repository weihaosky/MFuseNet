# pytorch
import torch, torchvision
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

# tools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, argparse, time, cv2
from PIL import Image

from network import FuseNetwork
from dataloader import MBdatalist
from dataloader import MBloader
from fuse_utils import train, test1, eval


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cost Fusing')
    parser.add_argument('--name', type=str, 
                        help='log name (default: "1")', default="1")
    parser.add_argument('--trainpath', default='dataset/TRAIN/',
                        help='training data set path')
    parser.add_argument('--testpath', default='dataset/TEST/',
                        help='test data set path')
    parser.add_argument('--evalpath', default='dataset/EVAL/',
                        help='evaluate data set path')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='maxium epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', type=int, default=0, 
                    help='if resume from previous model (default: Non)')
    parser.add_argument('--resume_model', default=None, 
                    help='previous model to resume (default: None)')
    parser.add_argument('--maxdisp', type=int, default=60,
                        help='maxium disparity')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    log_path = 'log/' + args.name + '/'
    args.output_path = log_path + "output/"
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_path + 'train/'):
        os.makedirs(output_path + 'train/')
    if not os.path.exists(output_path + 'test/'):
        os.makedirs(output_path + 'test/')
    if not os.path.exists(output_path + 'eval/'):
        os.makedirs(output_path + 'eval/')

    model = FuseNetwork(args.maxdisp, 1)
    model = nn.DataParallel(model, device_ids=[0, 1])
    if args.cuda:
        model = model.cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    epoch_begin = 0
    if args.resume:   
        epoch_begin = args.resume
        if args.resume_model:
            model.load_state_dict(torch.load(args.resume_model))
        else:
            model.load_state_dict(torch.load(log_path + 'model' + str(args.resume) + '.pth'))

    train_left_bin, train_right_bin, train_img, train_disp = MBdatalist.dataloader(args.trainpath, mode='3view')
    test_left_bin, test_right_bin, test_img, test_disp = MBdatalist.dataloader(args.testpath, mode='3view')

    TrainImgLoader = torch.utils.data.DataLoader(
         MBloader.myImageFloder([train_left_bin, train_right_bin], train_img, train_disp, mode='3view',
         dispRange=args.maxdisp, training=True, augment=False), 
         batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
         MBloader.myImageFloder([test_left_bin, test_right_bin], test_img, test_disp, mode='3view',
         dispRange=args.maxdisp, training=False, augment=False), 
         batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    print("Train dataset size:", len(train_left_bin))
    print("Test dataset size:", len(test_left_bin))

    # ========================= Optimization Setup ==============================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    train_loss_record = []
    test_record = []
    eval_record = []

    for epoch in range(epoch_begin+1, args.epochs+1):
        total_train_loss = 0
        total_test_loss = 0
        time_epoch = 0  

        ## ====================== training =======================
        for batch_idx, (costL_crop, costR_crop, img_crop, disp_crop) in enumerate(TrainImgLoader):
            start_time = time.time() 

            # normalization
            img_crop = img_crop.float() / 255.0

            loss = train(model, optimizer, [costR_crop], img_crop, disp_crop, args)
            total_train_loss += loss
            time_epoch += time.time() - start_time

        loss_mean = [total_train_loss/len(TrainImgLoader)]
        train_loss_record.append(loss_mean)

        plt.figure(0)
        plt.plot(range(1, len(train_loss_record) + 1), np.asarray(train_loss_record)[:, 0], color='blue', linewidth=0.5, linestyle='-')
        plt.title("loss--epoch")
        plt.legend(['train loss'])
        plt.savefig(log_path + "train_loss.png")

        print('epoch %d mean training loss = %.3f, one batch time = %.2f'
                % (epoch, loss_mean[0], time_epoch/len(TrainImgLoader)) )

        ## ======================== TEST ============================
        if epoch % 10 == 0:
            test_res = test1(model, args, epoch)
            test_record.append(test_res)
            plt.figure(1)
            plt.plot(range(1, len(test_record) + 1), np.asarray(test_record)[:, 0], color='blue', linewidth=0.5, linestyle='-')
            plt.plot(range(1, len(test_record) + 1), np.asarray(test_record)[:, 1], color='green', linewidth=0.5, linestyle='-')
            plt.title("loss--epoch")
            plt.legend(['test avgerr', 'test rms'])
            plt.savefig(log_path + "test_record.png")

        # ========================= SAVE ===================================
        if epoch % 50 == 0:
            torch.save(model.state_dict(), log_path + 'model' + str(epoch) + '.pth')
        