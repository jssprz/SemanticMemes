from random import randint
import torch
from torch.utils.data import Dataset, DataLoader

class MemeToTextDataset(Dataset):
    def __init__(self, embeddings, data, img_features, interval):
        self.img_features = img_features

        self.wordvectors = embeddings
        self.vocab = {}
        for k, v in data['vocab'].items():
            self.vocab[v] = k

        self.texts = []
        self.text_embeddings = []
        self.ids = []
        for i, (target, text, desc, interp) in enumerate(zip(data['images']['targets'], data['images']['texts'], data['images']['interpretations'], data['images']['descriptions'])):
            if i >= interval[0] and i <= interval[1] and (target == 1 or target == 4):
                if len(text):
                    self.ids.append(i)
                    self.text_embeddings.append(self.__text_emmbeding(text))
                    self.texts.append(self.__sentence(text))
                if len(desc):
                    self.ids.append(i)
                    self.text_embeddings.append(self.__text_emmbeding(desc))
                    self.texts.append(self.__sentence(desc))
                if len(interp):
                    self.ids.append(i)
                    self.text_embeddings.append(self.__text_emmbeding(interp))
                    self.texts.append(self.__sentence(interp))
    
    def __getitem__(self, i):
        idx = self.ids[i]
        n_i = randint(0, len(self.texts)-1)
        while self.ids[n_i] == idx:
            n_i = randint(0, len(self.texts)-1)
        return idx, self.img_features[idx], torch.FloatTensor(self.text_embeddings[i]), self.texts[i], torch.FloatTensor(self.text_embeddings[n_i]), self.texts[n_i]
    
    def __len__(self):
        return len(self.texts)
    
    def __sentence(self, text):
        words = []
        for t in text:
            words.append(self.vocab[t])
        return ' '.join(words)

    def __text_emmbeding(self, text):
        result = []
        for t in text:
            s = self.vocab[t]
            try:
                vec = self.wordvectors[s]
            except:
                # print('error with word "{}"'.format(s))
                pass
            else:
                result.append(torch.from_numpy(vec).view(1, -1))
        if len(result):
            return torch.mean(torch.cat(result, dim=0), dim=0)
        else:
            return torch.from_numpy(self.wordvectors['a'])

def get_train_loader(embeddings, data, img_features, batch_size):
    interval = (0, 49999)
    dset = MemeToTextDataset(embeddings, data, img_features, interval)
    return DataLoader(dset, batch_size, shuffle=True)

def get_test_loader(embeddings, data, img_features, batch_size):
    interval = (50000, 51999)
    dset = MemeToTextDataset(embeddings, data, img_features, interval)
    return DataLoader(dset, batch_size, shuffle=False)
