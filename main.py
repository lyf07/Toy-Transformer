import transformer
import torch
import torch.nn as nn
import random
import torch.optim as optim
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt


embed_dim = 512
src_vocab_size = tgt_vocab_size = 20
num_layers = 3
seq_len = 14
expansion_factor = 4
n_heads = 4

stoi = {'<bos>': 0, '<eos>': 1, '<pad>': 2}




def generate_random_batch(batch_size, max_len=12):
    src = []
    for _ in range(batch_size):
        random_len = random.randint(5, max_len - 2)
        random_seq = [stoi['<bos>']] + [random.randint(3, src_vocab_size - 1) for _ in range(random_len)] + [stoi['<eos>']]
        random_seq = random_seq + (max_len - random_len) * [stoi['<pad>']]
        src.append(random_seq)
    src = torch.tensor(src)
    tgt = src[:, :-1]
    tgt_y = src[:, 1:]
    n_tokens = (tgt_y != stoi['<pad>']).sum()
    return src, tgt, tgt_y, n_tokens

epochs = 200
batch_size = 2

model = transformer.Transformer(embed_dim, src_vocab_size, tgt_vocab_size, num_layers, expansion_factor, n_heads)

# model = nn.Transformer(embed_dim, n_heads, num_layers)


def main():
    '''
    Here we want to do a copy task. Eg: with input [0,1,2,3,4,5], we want the output to be [0,1,2,3,4]
    '''
    train(epochs)
    # see_data()
    draw_train()
    src = torch.tensor([[0, 4, 3, 4, 6, 8, 9, 9, 8, 1, 2, 2]])
    tgt = torch.tensor([[0]])
    src_len = src.shape[1]
    for _ in range(src_len):
        # print(tgt)
        output_token = model(src, tgt)
        output_token = output_token.argmax(dim=1)
        output_token = output_token[:,-1]
        tgt = torch.cat([tgt, output_token.unsqueeze(0)], dim=1)
        # print(tgt)
        if output_token == stoi['<eos>']:
            break
    print(tgt)

epoch_list = []
current_loss_list = []

def see_data():
    for _ in range(3):
        src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size)
        print(f'src = {src}')
        print(f'tgt = {tgt}')
        print(f'tgt_y = {tgt_y}')
        print('=============')

def draw_train():
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Graph')
    plt.plot(epoch_list, current_loss_list, color='b')
    plt.show()


def train(epochs):
    loss_func = nn.CrossEntropyLoss()
    adam = optim.Adam(model.parameters(), lr=1e-3)
    total_loss = 0
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        adam.zero_grad()
        src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size)
        # print(f'src = {src}, tgt = {tgt}, tgt_y = {tgt_y}, n_tokens = {n_tokens}')
        # model.train()
        output = model(src, tgt)
        # print(f'output = {output.contiguous().view(-1, output.size(-1))}')
        # print(f'tgt_y = {tgt_y.contiguous().view(-1)}')
        loss = loss_func(output.contiguous().view(-1, output.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
        loss.backward()
        adam.step()
        total_loss += loss.item()
        if epoch != 0 and epoch % 5 == 0:
            pbar.set_description(f'Epoch:{epoch}, average loss: {total_loss / epoch}, current loss: {loss.item()}')
            epoch_list.append(epoch)
            current_loss_list.append(loss.item())

main()
