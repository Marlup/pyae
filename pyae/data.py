from torch.utils.data import Dataset
from torch import rand_like

class EMIDataset(Dataset):
    def __init__(self, x, x_categories=None, ids=None, target_feature_index=0, noise=0.0):
        self.x = x
        self.x_categories = x_categories
        self.ids = ids
        self.target_feature_index = target_feature_index
        self.noise = noise
    
    def __getitem__(self, index):
        data_output = {}

        if self.noise > 0.0:
            x = self.x[index] + self.noise * rand_like(self.x[index])
        else:
            x = self.x[index]
            
        data_output.update({"x": x})
        
        if self.x_categories is not None:
            data_output.update({"x_categories": self.x_categories[index]})
        
        if len(self.x.shape) == 3:
            y = self.x[index, [self.target_feature_index]] # to keep dims
        elif len(self.x.shape) == 2:
            y = self.x[index]
        else:
            y = self.x[index]
        
        data_output.update({"y": y})
    
        if self.ids is not None:
            data_output.update({"ids": self.ids[index]})
        else:
            data_output.update({"ids": ()})
        
        return data_output
    
    def __len__(self):
        return len(self.x)

class EMIDatasetClassifier(Dataset):
    def __init__(self, x, y, x_categories=None, ids=None, noise=0.0, include_ids=True):
        self.x = x
        self.y = y
        self.x_categories = x_categories
        self.ids = ids
        self.noise = noise
    
    def __getitem__(self, index):
        data_output = {}

        # Get and add x tensor
        if self.noise > 0.0:
            x = self.x[index] + self.noise * rand_like(self.x[index])
        else:
            x = self.x[index]
            
        data_output.update({"x": x})
        
        # Get and add y tensor
        y = self.y[index]
        data_output.update({"y": y})
        
        # Get and add x_categories tensor
        if self.x_categories is not None:
            data_output.update({"x_categories": self.x_categories[index]})
            
        # Get and add ids tensor
        if include_ids and self.ids is not None:
            data_output.update({"ids": self.ids[index]})
        elif include_ids:
            data_output.update({"ids": ()})
        
        return data_output
    
    def __len__(self):
        return len(self.x)