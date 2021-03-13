import wget, pickle, os

def load_data():
    # Load dataset from DIYA GitLab
    url = "https://gitlab.diyaml.com/moong1234/application/raw/release/data.pkl"
    if not os.path.isfile("data.pkl"):
        wget.download(url, 'data.pkl')
    
    path = "data.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

data, label = load_data()
print(data.shape, label.shape)
data, label





import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO
from sklearn.model_selection import train_test_split

data1, data2 = data[0:768], data[768:]
label1, label2 = label[0:768], label[768:]

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data2, label2, train_size = 0.8, test_size = 0.2)

validation_data_x = data1

validation_data_y =label1

train_data_x = torch.tensor(train_data_x, dtype=torch.float64, device=device).double()
train_data_y = torch.tensor(train_data_y, dtype=torch.float64, device=device).double()
test_data_x = torch.tensor(test_data_x, dtype=torch.float64, device=device).double()
test_data_y = torch.tensor(test_data_y, dtype=torch.float64, device=device).double()
validation_data_x = torch.tensor(validation_data_x, dtype=torch.float64, device=device).double()
validation_data_y = torch.tensor(validation_data_y, dtype=torch.float64, device=device).double()




from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(train_data_x, train_data_y)
# TODO
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)




class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO
        self.hello = 'test'

    def forward(self, x):
        # TODO
        self.hi = 'test2'











