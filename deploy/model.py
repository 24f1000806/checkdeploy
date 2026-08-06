import torch.nn as nn
import torch.nn.functional as F


class BiLSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                                     
        self.lstm = nn.LSTM( embed_dim, hidden_dim,
                             batch_first=True, bidirectional=True )

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.embedding(x)
        output, _ = self.lstm(x)
        x = output.mean(dim=1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)

        return x
