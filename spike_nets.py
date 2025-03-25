import torch
import snntorch as snn

def to_poisson_spikes(data, steps: int, max_rate: int=200):
    """generates poisson spike trains"""
    # Rescale factor for Poisson distribution
    rescale_factor = max_rate / steps
    rand_vals = torch.rand(steps, *data.shape, device=data.device)

    # Compare against intensity to generate spikes
    spikes = (rand_vals < data * rescale_factor).float()
    
    return spikes

class RW_Module(torch.nn.Module):
    """an empty to subclass saver and loader functions"""
    def __init__(self):
        super().__init__()

    def save_parameters(self, path: str):
        torch.save(self.in_layer, path + "0.pt")
        torch.save(self.h1_layer, path + "1.pt")
        torch.save(self.h2_layer, path + "2.pt")

    def load_parameters(self, path: str):
        self.in_layer = torch.load(path + "0.pt", weights_only=False)
        self.h1_layer = torch.load(path + "1.pt", weights_only=False)
        self.h2_layer = torch.load(path + "2.pt", weights_only=False)

class FC_Net(RW_Module):
    """a simple fully connected ReLU activated neural network of just 3 layers"""
    def __init__(self, n_x: int, n_h: list, n_y: int):
        super().__init__()

        self.in_layer = torch.nn.Linear(n_x, n_h[0], bias=False)
        self.h1_layer = torch.nn.Linear(n_h[0], n_h[1], bias=False)
        self.h2_layer = torch.nn.Linear(n_h[1], n_y, bias=False)
        self.dropout = torch.nn.Dropout()
        self.activator = torch.nn.ReLU()

    def forward(self, x):
        # Flatten images
        x = x.view(x.size(0), -1)
        
        inp = self.dropout(self.activator(self.in_layer(x)))
        h1 = self.dropout(self.activator(self.h1_layer(inp)))
        y = self.activator(self.h2_layer(h1))

        return y

class FC_SNN(RW_Module):
    """a fully connected spiking neural network. identical to FC_Net but with IF neurons."""
    def __init__(self, n_x: int, n_h: list, n_y: int, beta: float=0, threshold: float=1, steps: int=100, rate: int=200):
        super().__init__()

        self.in_layer = torch.nn.Linear(n_x, n_h[0], bias=False)
        self.h1_layer = torch.nn.Linear(n_h[0], n_h[1], bias=False)
        self.h2_layer = torch.nn.Linear(n_h[1], n_y, bias=False)
        self.in_active = snn.Leaky(beta=beta, threshold=threshold)
        self.h1_active = snn.Leaky(beta=beta, threshold=threshold)
        self.h2_active = snn.Leaky(beta=beta, threshold=threshold)

        self.steps = steps
        self.rate = rate

    def forward(self, x):
        # Flatten images
        x = x.view(x.size(0), -1)
        x = to_poisson_spikes(x, self.steps, self.rate)
        # x = snn.spikegen.rate(x, self.steps)

        memin = self.in_active.reset_mem()
        memh1 = self.h1_active.reset_mem()
        memh2 = self.h2_active.reset_mem()

        out_spikes = []
        memh2_mem = []

        for step in x:
            curin = self.in_layer(step)
            spkin, memin = self.in_active(curin, memin)
            curh1 = self.h1_layer(spkin)
            spkh1, memh1 = self.h1_active(curh1, memh1)
            curh2 = self.h2_layer(spkh1)
            spkh2, memh2 = self.h2_active(curh2, memh2)

            out_spikes.append(spkh2)
            memh2_mem.append(memh2)

        return torch.stack(out_spikes), torch.stack(memh2_mem)

class FC_Count_Net(RW_Module):
    """a counting variant of FC_Net used for data-based normalisation."""
    def __init__(self, n_x: int, n_h: list, n_y: int):
        super().__init__()

        self.dims = [n_x, n_h, n_y]

        self.in_layer = torch.nn.Linear(n_x, n_h[0], bias=False)
        self.h1_layer = torch.nn.Linear(n_h[0], n_h[1], bias=False)
        self.h2_layer = torch.nn.Linear(n_h[1], n_y, bias=False)
        self.activator = torch.nn.ReLU()

        # to store maximum activations
        self.maxin_act = torch.zeros([n_h[0]])
        self.maxh1_act = torch.zeros([n_h[1]])
        self.maxh2_act = torch.zeros([n_y])

    def forward(self, x):
        # Flatten images
        x = x.view(x.size(0), -1)
        
        inp = self.activator(self.in_layer(x))
        self.maxin_act = torch.maximum(self.maxin_act, inp)

        h1 = self.activator(self.h1_layer(inp))
        self.maxh1_act = torch.maximum(self.maxh1_act, h1)

        y = self.activator(self.h2_layer(h1))
        self.maxh2_act = torch.maximum(self.maxh2_act, y)

        return y
    
    def reset_max_count(self):
        """reset maximum activation memory"""
        self.maxin_act = torch.zeros([self.dims[1][0]])
        self.maxh1_act = torch.zeros([self.dims[1][1]])
        self.maxh2_act = torch.zeros([self.dims[2]])

class Conv_Net(RW_Module):
    """a simple convolutional neural network with 2 conv layers and 1 FC layer."""
    def __init__(self, n_x: int, n_h: list, n_y: list, kernel_size: int=5):
        super().__init__()

        self.in_layer = torch.nn.Conv2d(n_x, n_h[0], kernel_size, bias=False)
        self.h1_layer = torch.nn.Conv2d(n_h[0], n_h[1], kernel_size, bias=False)
        self.h2_layer = torch.nn.Linear(n_y[0], n_y[1], bias=False)
        self.pooling = torch.nn.AvgPool2d(2)
        self.dropout = torch.nn.Dropout()
        self.activator = torch.nn.ReLU()

    def forward(self, x):
        inp = self.dropout(self.activator(self.pooling(self.in_layer(x))))
        h1 = self.dropout(self.activator(self.pooling(self.h1_layer(inp))))
        
        # vectorise image
        h1 = h1.view(h1.size(0), -1)
        y = self.activator(self.h2_layer(h1))

        return y

class Conv_SNN(RW_Module):
    """a Conv_Net but with IF neurons instead of ReLU activators"""
    def __init__(self, n_x: int, n_h: list, n_y: list, kernel_size: int=5, beta: float=0, threshold: float=1, steps: int=100, rate: int=200):
        super().__init__()

        self.in_layer = torch.nn.Conv2d(n_x, n_h[0], kernel_size, bias=False)
        self.h1_layer = torch.nn.Conv2d(n_h[0], n_h[1], kernel_size, bias=False)
        self.h2_layer = torch.nn.Linear(n_y[0], n_y[1], bias=False)
        self.pooling = torch.nn.AvgPool2d(2)
        self.in_active = snn.Leaky(beta=beta, threshold=threshold)
        self.h1_active = snn.Leaky(beta=beta, threshold=threshold)
        self.h2_active = snn.Leaky(beta=beta, threshold=threshold)

        self.steps = steps
        self.rate = rate

    def forward(self, x):
        x = to_poisson_spikes(x, self.steps, self.rate)

        memin = self.in_active.reset_mem()
        memh1 = self.h1_active.reset_mem()
        memh2 = self.h2_active.reset_mem()

        out_spikes = []
        memh2_mem = []

        for step in x:
            curin = self.pooling(self.in_layer(step))
            spkin, memin = self.in_active(curin, memin)
            curh1 = self.pooling(self.h1_layer(spkin))
            spkh1, memh1 = self.h1_active(curh1, memh1)

            # vectorise spike image
            spkh1 = spkh1.view(spkh1.size(0), -1)
            
            curh2 = self.h2_layer(spkh1)
            spkh2, memh2 = self.h2_active(curh2, memh2)

            out_spikes.append(spkh2)
            memh2_mem.append(memh2)

        return torch.stack(out_spikes), torch.stack(memh2_mem)

class Conv_Count_Net(RW_Module):
    """a counting variant of Conv_Net used for data-based normalisation."""
    def __init__(self, n_x: int, n_h: list, n_y: list, kernel_size: int=5, image_size: list=(28, 28)):
        super().__init__()

        self.in_layer = torch.nn.Conv2d(n_x, n_h[0], kernel_size, bias=False)
        self.h1_layer = torch.nn.Conv2d(n_h[0], n_h[1], kernel_size, bias=False)
        self.h2_layer = torch.nn.Linear(n_y[0], n_y[1], bias=False)
        self.pooling = torch.nn.AvgPool2d(2)
        self.activator = torch.nn.ReLU()

        # to store maximum activations
        size_in_x = (image_size[0] - kernel_size + 1) // 2
        size_in_y = (image_size[1] - kernel_size + 1) // 2
        self.maxin_act = torch.zeros([n_h[0], size_in_x, size_in_y])
        size_h1_x = (size_in_x - kernel_size + 1) // 2
        size_h1_y = (size_in_y - kernel_size + 1) // 2
        self.maxh1_act = torch.zeros([n_h[1], size_h1_x, size_h1_y])
        self.maxh2_act = torch.zeros([n_y[1]])

    def forward(self, x):
        
        inp = self.activator(self.pooling(self.in_layer(x)))
        self.maxin_act = torch.maximum(self.maxin_act, inp)
        h1 = self.activator(self.pooling(self.h1_layer(inp)))
        self.maxh1_act = torch.maximum(self.maxh1_act, h1)
        
        # vectorise image
        h1 = h1.view(h1.size(0), -1)
        y = self.activator(self.h2_layer(h1))
        self.maxh2_act = torch.maximum(self.maxh2_act, y)

        return y