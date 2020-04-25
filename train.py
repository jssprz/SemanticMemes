import os
import sys
import numpy as np

import torch
from torch.autograd import Variable
from sklearn.metrics import average_precision_score

from model import MLP
from loss import TripletLoss
from data_loader import get_train_loader, get_test_loader


def train(visual_encoder, train_loader, test_loader, criterion, optimizer):
    save_checkpoints_dir = os.path.join('./models')

    train_losses, test_losses, maps = [], [], []
    last_saved_epoch, early_stopping, max_map = -1, 0, 0
    for e in range(100000):
    #     text_encoder.train()
        visual_encoder.train()
        total_train_loss = 0
        for i, (ids, v_feats, p_t_feats, p_texts, n_t_feats, n_texts) in enumerate(train_loader):
            p_t_feats = Variable(p_t_feats)
            n_t_feats = Variable(n_t_feats)
            v_feats = Variable(v_feats)
                    
    #         p_t_enc = text_encoder(p_t_feats)
    #         n_t_enc = text_encoder(n_t_feats)
            v_enc = visual_encoder(v_feats)

            loss = criterion(v_enc, p_t_feats, n_t_feats)
            total_train_loss += loss.item()
            
            sys.stdout.write('\r epoch: {0:03d} iter: {1:03d} train-loss: {2:.5f}'.format(e, i, loss.item()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_losses.append(total_train_loss/len(train_loader))        
        print('')
    #     text_encoder.eval()
        visual_encoder.eval()
        
        ids, v_feats, p_t_feats, p_texts, n_t_feats, n_texts = list(test_loader)[0]
        p_t_feats = Variable(p_t_feats)
        n_t_feats = Variable(n_t_feats)
        v_feats = Variable(v_feats)
        
        with torch.no_grad():
    #         p_t_enc = text_encoder(p_t_feats)
    #         n_t_enc = text_encoder(n_t_feats)
            v_enc = visual_encoder(v_feats)

        loss = criterion(v_enc, p_t_feats, n_t_feats)
        test_losses.append(loss.item())
        
        total_map_score = 0
        last_idx = -1
        q_count = 0
        for i in range(v_enc.size(0)):
            if ids[i] != last_idx:
                last_idx = ids[i]
                q_count += 1
                dists = torch.sum((v_enc[i, :] - p_t_feats)**2, dim=1)
    #             dists = torch.rand(v_enc.size(0)) * 20

                y_true = np.array([1 if idx == ids[i] else 0 for idx in ids])
                total_map_score += average_precision_score(y_true, -1 * dists.numpy())

        mAP = total_map_score/q_count
        maps.append(mAP)
        
        early_stopping += 1
        if mAP > max_map:
            early_stopping = 0
            max_map = mAP

            torch.save(obj={'epoch': e + 1,
                            'visual_encoder': visual_encoder.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'map': mAP},
                        f=os.path.join(save_checkpoints_dir, 'chkpt_{}.pkl'.format(e)))

            # remove previously saved
            if last_saved_epoch != -1:
                os.remove(os.path.join(save_checkpoints_dir,'chkpt_{}.pkl'.format(last_saved_epoch)))
            last_saved_epoch = e
        
        sys.stdout.write('\r epoch: {0:03d} avg-train-loss: {1:.5f} test-loss: {2:.5f} mAP: {3:.3f}\n'.format(e, total_train_loss/len(train_loader), loss.item(), mAP))
        
        if early_stopping == 100:
            print('Early stopping at epoch {}'.format(e))
            break

def init_laders(batch_size):
    import json
    import h5py
    from gensim.models.keyedvectors import KeyedVectors

    with open('./data/datainfo-v1.0.json', 'r') as f:
        data = json.load(f)

    f = h5py.File('./data/resnet_features.hdf5', 'r')
    img_features = f['resnet152_features'][()]
    f.close()

    wordvectors_file_vec = './data/fasttext-sbwc.vec'
    count = 100000
    wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=count)

    train_loader = get_train_loader(wordvectors, data, img_features, batch_size)
    test_loader = get_test_loader(wordvectors, data, img_features, 200)

    return train_loader, test_loader

if __name__ == '__main__':
    # text_encoder = MLP(300, 1024, 128)
    visual_encoder = MLP(2048, 4096, 300)

    # optimizer = torch.optim.SGD([{'params': text_encoder.parameters()}, 
    #                              {'params': visual_encoder.parameters()}], lr=0.001)
    optimizer = torch.optim.SGD(visual_encoder.parameters(), lr=0.0001)

    # criterion = nn.MSELoss()
    criterion = TripletLoss(margin=2.0)

    # pdist = nn.PairwiseDistance(p=2)

    train_loader, test_loader = init_laders(batch_size=16)

    print(len(train_loader.dataset), len(test_loader.dataset))

    train(visual_encoder, train_loader, test_loader, criterion, optimizer)