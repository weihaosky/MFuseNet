import torch, torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, cv2

# Disp_type = ''    # the ground-truth disparity format is disparity*1
Disp_type = 'x4'    # the ground-truth disparity format is disparity*4

def train(model, optimizer, costs_input, img, disp, args, Test=False):

    if not Test:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    costs = []
    if args.cuda:
        img, disp = img.cuda(), disp.cuda()
        for cost in costs_input:
            costs.append(cost.cuda())
    
    output = model(costs)

    output = output.squeeze(1)
    mask = (disp != 0)

    loss = F.smooth_l1_loss(output[mask], disp[mask], reduction='mean')

    if not Test:
        loss.backward()
        optimizer.step()

    cv2.imwrite(args.output_path + 'train/img.png', 
                                255*img[0].permute([1,2,0]).detach().cpu().numpy())
    cv2.imwrite(args.output_path + 'train/outdisp.png', 
                                output[0].detach().cpu().numpy())

    return loss.data.item()


def test(model, args, epoch):
    test_record = []
    for scene in os.listdir(args.testpath):
        left_path = args.testpath + scene + '/' + 'left.bin'
        right_path = args.testpath + scene + '/' + 'right.bin'
        img_path = args.testpath + scene + '/' + 'view1.png'
        disp_path = args.testpath + scene + '/' + 'disp1' + Disp_type + '.png'
        
        d = args.maxdisp
        img = Image.open(img_path)
        w, h = img.size
        img = np.array(img).transpose(2, 0, 1)

        disp = Image.open(disp_path)
        disp = np.expand_dims(np.array(disp), 0)              

        left_mem = np.memmap(left_path, dtype=np.float32, shape=(1, d, h, w))
        right_mem = np.memmap(right_path, dtype=np.float32, shape=(1, d, h, w))  
        costL = np.squeeze(np.array(left_mem))
        costR = np.squeeze(np.array(right_mem))
        costL[np.isnan(costL)]=20
        costR[np.isnan(costR)]=20
        costL = torch.from_numpy(costL).unsqueeze(0).cuda()
        costR = torch.from_numpy(costR).unsqueeze(0).cuda()

        # pad to 16
        pad_h = (h / 16 + (1 if h % 16 != 0 else 0)) * 16 - h
        pad_w = (w / 16 + (1 if h % 16 != 0 else 0)) * 16 - w
        costL = F.pad(costL, (0, pad_w, 0, pad_h))
        costR = F.pad(costR, (0, pad_w, 0, pad_h))

        # for large image, process piece by piece
        SIZE = 512 
        Edge = 8
        wseg = w // SIZE + 1 if( w % SIZE != 0) else 0
        hseg = h // SIZE + 1 if( h % SIZE != 0) else 0
        outdisp = torch.ones(1, h+pad_h, w+pad_w).float().cuda()
        with torch.no_grad():
            for i in range(hseg):
                for j in range(wseg):
                    y1 = max(i*SIZE-Edge, 0)
                    y2 = (i+1)*SIZE + Edge
                    x1 = max(j*SIZE-Edge, 0)
                    x2 = (j+1)*SIZE + Edge
                    outdisp[:, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE] = \
                        model([costL[:, :, y1:y2, x1:x2], costR[:, :, y1:y2, x1:x2]]) \
                        [:,min(i,1)*Edge:min(i,1)*Edge+SIZE,min(j,1)*Edge:min(j,1)*Edge+SIZE]
            outdisp = outdisp[:, :h, :w]

            dispgt = torch.from_numpy(disp).float().cuda() 
            if(Disp_type == 'x4'):    # the ground-truth is disparity*4
                dispgt /= 4.0
            mask = (dispgt != 0)
            diff = torch.abs(outdisp[mask] - dispgt[mask])
            avgerr = torch.mean(diff)
            rms = torch.sqrt( (diff**2).mean() ) 
            bad05 = len(diff[diff>0.5])/float(len(diff))
            bad1 = len(diff[diff>1])/float(len(diff))
            bad2 = len(diff[diff>2])/float(len(diff))
        test_record.append([avgerr, rms, bad05, bad1, bad2])

        cv2.imwrite(args.output_path + 'test/' + scene + "_outdisp.png", 
                        outdisp.cpu().numpy().squeeze())
    test_res = np.array(test_record).mean(0)
    print('==== epoch %d test avgerr = %.3f, rms = %.3f, bad05 = %.3f, bad1 = %.3f, bad2 = %.3f ==='
                    % (epoch, test_res[0], test_res[1], test_res[2], test_res[3], test_res[4]) )
    return test_res


def eval(model, args, epoch):
    eval_record = []
    for scene in os.listdir(args.evalpath):
        left_path = args.evalpath + scene + '/' + 'left.bin'
        right_path = args.evalpath + scene + '/' + 'right.bin'
        img_path = args.evalpath + scene + '/' + 'view1.png'
        
        d = args.maxdisp
        img = Image.open(img_path)
        w, h = img.size             

        left_mem = np.memmap(left_path, dtype=np.float32, shape=(1, d, h, w))
        right_mem = np.memmap(right_path, dtype=np.float32, shape=(1, d, h, w))  
        costL = np.squeeze(np.array(left_mem))
        costR = np.squeeze(np.array(right_mem))
        costL[np.isnan(costL)]=20
        costR[np.isnan(costR)]=20
        costL = torch.from_numpy(costL).unsqueeze(0).cuda()
        costR = torch.from_numpy(costR).unsqueeze(0).cuda()

        # pad to 16
        pad_h = (h / 16 + (1 if h % 16 != 0 else 0)) * 16 - h
        pad_w = (w / 16 + (1 if h % 16 != 0 else 0)) * 16 - w
        costL = F.pad(costL, (0, pad_w, 0, pad_h))
        costR = F.pad(costR, (0, pad_w, 0, pad_h))
        
        # for large image
        SIZE = 512
        wseg = w // SIZE + 1 if( w % SIZE != 0) else 0
        hseg = w // SIZE + 1 if( h % SIZE != 0) else 0
        outdisp = torch.ones(1, h+pad_h, w+pad_w).float().cuda()
        with torch.no_grad():
            for i in range(hseg):
                for j in range(wseg):
                    outdisp[:, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE] = \
                        model([costL[:, :, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE], 
                                costR[:, :, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE]])
            outdisp = outdisp[:, :h, :w]
            
        cv2.imwrite(args.output_path + 'eval/' + scene + "_outdisp.png", 
                        outdisp.cpu().numpy().squeeze())

    return eval_record


def test4(model, args, epoch):
    # ====================== train data ========================
    test_record = []
    for scene in os.listdir(args.trainpath):
        left_path = args.trainpath + scene + '/left.bin'
        right_path = args.trainpath + scene + '/right.bin'
        bottom_path = args.trainpath + scene + '/bottom.bin'
        top_path = args.trainpath + scene + '/top.bin'
        img_path = args.trainpath + scene + '/view1.png'
        disp_path = args.trainpath + scene + '/disp1' + Disp_type + '.png'
        
        d = args.maxdisp
        img = Image.open(img_path)
        w, h = img.size
        img = np.array(img).transpose(2, 0, 1)

        disp = Image.open(disp_path)
        disp = np.expand_dims(np.array(disp), 0)              

        left_mem = np.memmap(left_path, dtype=np.float32, shape=(1, d, h, w))
        right_mem = np.memmap(right_path, dtype=np.float32, shape=(1, d, h, w))  
        costL = np.squeeze(np.array(left_mem))
        costR = np.squeeze(np.array(right_mem))
        costL[np.isnan(costL)]=20
        costR[np.isnan(costR)]=20
        costL = torch.from_numpy(costL).unsqueeze(0).cuda()
        costR = torch.from_numpy(costR).unsqueeze(0).cuda()

        bottom = np.memmap(bottom_path, dtype=np.float32, shape=(1, d, w, h))
        top = np.memmap(top_path, dtype=np.float32, shape=(1, d, w, h))
        bottom=np.rot90(np.array(bottom), k=-1, axes=(2,3)).copy()
        top=np.rot90(np.array(top), k=-1, axes=(2,3)).copy()
        bottom[np.isnan(bottom)]=20
        top[np.isnan(top)]=20
        bottom = np.squeeze(bottom)
        top = np.squeeze(top)
        costB = torch.from_numpy(bottom).unsqueeze(0).cuda()
        costT = torch.from_numpy(top).unsqueeze(0).cuda()

        # for large image
        SIZE = 512
        wseg = w // SIZE + 1 if( w % SIZE != 0) else 0
        hseg = w // SIZE + 1 if( h % SIZE != 0) else 0
        outdisp = torch.ones(1, h, w).float().cuda()
        with torch.no_grad():
            for i in range(hseg):
                for j in range(wseg):
                    outdisp[:, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE] = \
                        model([costL[:, :, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE], 
                               costR[:, :, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE],
                               costB[:, :, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE],
                               costT[:, :, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE]])

            outdisp = outdisp.squeeze(1)
            dispgt = torch.from_numpy(disp).float().cuda() 
            if(Disp_type == 'x4'):  # the ground-truth is disparity*4
                dispgt /= 4.0   
            mask = (dispgt != 0)
            diff = torch.abs(outdisp[mask] - dispgt[mask])
            avgerr = torch.mean(diff)
            rms = torch.sqrt( (diff**2).mean()) 
            bad05 = len(diff[diff>0.5])/float(len(diff))
            bad1 = len(diff[diff>1])/float(len(diff))
            bad2 = len(diff[diff>2])/float(len(diff))
            
        test_record.append([avgerr, rms, bad05, bad1, bad2])

        cv2.imwrite(args.output_path + 'test/' + scene + "_outdisp.png", 
                        outdisp.cpu().numpy().squeeze())
    test_res = np.array(test_record).mean(0)
    print('======= epoch %d avgerr = %.3f, rms = %.3f, bad05 = %.3f, bad1 = %.3f, bad2 = %.3f ======='
                    % (epoch, test_res[0], test_res[1], test_res[2], test_res[3], test_res[4]))
    return test_res


def test1(model, args, epoch):
    test_record = []
    for scene in os.listdir(args.testpath):
        right_path = args.testpath + scene + '/' + 'right.bin'
        img_path = args.testpath + scene + '/' + 'view1.png'
        disp_path = args.testpath + scene + '/' + 'disp1' + Disp_type + '.png'
        
        d = args.maxdisp
        img = Image.open(img_path)
        w, h = img.size
        img = np.array(img).transpose(2, 0, 1)

        disp = Image.open(disp_path)
        disp = np.expand_dims(np.array(disp), 0)              

        right_mem = np.memmap(right_path, dtype=np.float32, shape=(1, d, h, w))  
        costR = np.squeeze(np.array(right_mem))
        costR[np.isnan(costR)]=20
        costR = torch.from_numpy(costR).unsqueeze(0).cuda()

        # pad to 16
        pad_h = (h / 16 + (1 if h % 16 != 0 else 0)) * 16 - h
        pad_w = (w / 16 + (1 if h % 16 != 0 else 0)) * 16 - w
        costR = F.pad(costR, (0, pad_w, 0, pad_h))

        # for large image
        SIZE = 512 
        Edge = 8
        wseg = w // SIZE + 1 if( w % SIZE != 0) else 0
        hseg = h // SIZE + 1 if( h % SIZE != 0) else 0
        outdisp = torch.ones(1, h+pad_h, w+pad_w).float().cuda()
        with torch.no_grad():
            for i in range(hseg):
                for j in range(wseg):
                    y1 = max(i*SIZE-Edge, 0)
                    y2 = (i+1)*SIZE + Edge
                    x1 = max(j*SIZE-Edge, 0)
                    x2 = (j+1)*SIZE + Edge
                    outdisp[:, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE] = \
                        model([costR[:, :, y1:y2, x1:x2]]) \
                        [:,min(i,1)*Edge:min(i,1)*Edge+SIZE,min(j,1)*Edge:min(j,1)*Edge+SIZE]
            outdisp = outdisp[:, :h, :w]

            dispgt = torch.from_numpy(disp).float().cuda() 
            if(Disp_type == 'x4'):  # the ground-truth is disparity*4
                dispgt /= 4.0
            mask = (dispgt != 0)
            diff = torch.abs(outdisp[mask] - dispgt[mask])
            avgerr = torch.mean(diff)
            rms = torch.sqrt( (diff**2).mean() ) 
            bad05 = len(diff[diff>0.5])/float(len(diff))
            bad1 = len(diff[diff>1])/float(len(diff))
            bad2 = len(diff[diff>2])/float(len(diff))
        test_record.append([avgerr, rms, bad05, bad1, bad2])

        cv2.imwrite(args.output_path + 'test/' + scene + "_outdisp.png", 
                        outdisp.cpu().numpy().squeeze())
    test_res = np.array(test_record).mean(0)
    print('==== epoch %d test avgerr = %.3f, rms = %.3f, bad05 = %.3f, bad1 = %.3f, bad2 = %.3f ==='
                    % (epoch, test_res[0], test_res[1], test_res[2], test_res[3], test_res[4]) )
    return test_res