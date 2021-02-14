from PIL import Image

import transformers

import torch

import torchvision
import torchvision.transforms as T
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, \
    pad_sequence

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.inception import Inception3
from warnings import warn


class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else:
            warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x


from torch.utils.model_zoo import load_url


def beheaded_inception_v3(transform_input=True):
    model = BeheadedInception3(transform_input=transform_input)
    inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
    model.load_state_dict(load_url(inception_url))
    return model


class Encoder(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.fc = nn.Linear(2048, emb_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, caption, features):
        embedded = F.dropout2d(self.embedding(caption), 0.5).squeeze(2)
        # [bs, seq_len - 1, emb_size]
        input_ = torch.cat([features.unsqueeze(1), embedded[:, :-1, :]], dim=1)
        # input_ = self.attention(features, input_)
        output, lstm_state = self.lstm(input_)

        return self.linear(output)

    def sample(self, features, len_=30):
        outputs = torch.zeros(features.size(0), len_, self.vocab_size)

        lstm_state = None

        input_ = features.unsqueeze(1)
        for i in range(len_):
            # input_ = self.attention(features, input_)
            output, lstm_state = self.lstm(input_, lstm_state)
            output = self.linear(output)

            outputs[:, i, :] = output

            max_pick = output.max(2)[1]
            input_ = F.dropout2d(self.embedding(max_pick), 0.5)

        return outputs


class Model(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size):
        super().__init__()

        self.encoder = Encoder(emb_size)
        self.decoder = Decoder(emb_size, hidden_size, vocab_size)

    def forward(self, img, captions=None, len_=30):
        features = self.encoder(img)
        # features shape: [bs, emb_size]
        if captions is not None:
            captions = captions.permute(1, 0).unsqueeze(2)
            # [bs, len_seq, 1]
            output = self.decoder(captions, features)
        else:
            output = self.decoder.sample(features)

        return output


class nnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.download_weights()
        path_pretrained_model = 'static/weights/model.pth'
        EMBEDDING_SIZE, HIDDEN_SIZE, VOCAB_SIZE = 384, 512, 50260

        print('download inception')
        self.encoder = beheaded_inception_v3().train(False)
        print('succeed!')

        self.model = Model(EMBEDDING_SIZE, HIDDEN_SIZE, VOCAB_SIZE)
        print('load state dict')
        self.model.load_state_dict(torch.load(path_pretrained_model, map_location=torch.device('cpu')))

    def download_weights(self):
        import requests
        from urllib.parse import urlencode

        base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
        public_key = 'https://disk.yandex.ru/d/bv0wD8_JzYb2fQ'  # Сюда вписываете вашу ссылку

        # Получаем загрузочную ссылку
        final_url = base_url + urlencode(dict(public_key=public_key))
        response = requests.get(final_url)
        download_url = response.json()['href']

        # Загружаем файл и сохраняем его
        download_response = requests.get(download_url)
        with open('static/weights/model.pth', 'wb') as f:  # Здесь укажите нужный путь к файлу
            f.write(download_response.content)

    def forward(self, img):
        features = self.encoder(img)[1]
        out = self.model(features)

        return out


class Model_():
    def __init__(self):

        print('create model')
        self.model = nnModel()

        print('model creates!!!')
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

        self.tr = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor()
        ])

    def __call__(self, path):
        pil_img = Image.open(path).convert('RGB')
        torch_img = self.tr(pil_img)

        logits = self.model(torch_img.unsqueeze(0)).detach()
        output = logits.max(2)[1]
        sentence = self.tokenizer.decode(output[0])
        return sentence
