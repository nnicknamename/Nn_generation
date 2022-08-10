from torch.utils.data import Dataset
import torch
import random 
class MixableDataset(Dataset):
    r"""
    mixable dataset takes a list of :class:`Dataset` and combines them into one dataset
    """
    def __init__(self,datasets):
        super().__init__()
        self.datasets=datasets 
        self.init_Classes()
        self.extract_targets()
        self.calculate_length()
        self.idx_table=[[i for i,x in enumerate(self.targets==c) if x==True] for c in range(len(self.classes))]

    def extract_targets(self):
        targets=[] 
        for dataset in self.datasets:
            targets.extend(map(lambda x: self.transform_label(x,dataset),dataset.targets))
        self.targets=torch.Tensor(targets)
    def init_Classes(self):
        self.classes=[]
        for dataset in self.datasets:
            self.classes.extend(dataset.classes) 

    def calculate_length(self):
        length=0
        for dataset in self.datasets:
            length+=dataset.__len__()
        self.length=length

    def __len__(self):
        return self.length

    def map_index(self,index):
        len=0
        for dataset in self.datasets:
            len+=dataset.__len__()
            if len >index:
                return dataset, index-(len-dataset.__len__())
    
    def transform_label(self,label,dataset):
        return self.classes.index(dataset.classes[label])

    #TODO: implement the transformation (its not important for now).
    def transform_datapoint(self,datapoint):
        return datapoint

    def __getitem__(self, index):
        dataset,index=self.map_index(index)
        datapoint,label=dataset.__getitem__(index)
        return self.transform_datapoint(datapoint),self.transform_label(label,dataset)

    def get_subDatast(self,focusClass,size=None):
        if size==None:
            dataset_idx=self.idx_table[focusClass].copy()
        else:
            assert size<=len(self.idx_table[focusClass]) , 'the given size is larger than the focus class'
            dataset_idx=random.sample(self.idx_table[focusClass],size)
        #dataset_idx=self.idx_table[focusClass].copy()
        
        for i in [i for i in range(len(self.classes)) if i != focusClass] :
            s=int(len(dataset_idx)/(len(self.classes)))
            if s>len(self.idx_table[i]):
                s=len(self.idx_table[i])
                dataset_idx.extend(self.idx_table[i])
            dataset_idx.extend(random.sample(self.idx_table[i],s))
        return Subdataset(self,dataset_idx,focusClass)


class Subdataset(Dataset):
    """
    a :class:`Subdataset` is a dataset scentered around a class from a given dataset, this class is called a focusClass, where half the data is 
        in the focus class the rest is a mix of the rest of the classes.
    """
    def __init__(self,dataset,idx_table,focusClass):
        super().__init__()
        self.dataset=dataset
        self.idx_table=idx_table
        self.focusClass=focusClass
        self.calculate_targets()
        
    def calculate_targets(self):
        targets=[]
        for i in self.idx_table:
            targets.append(self.dataset.targets[i])
        self.targets=torch.Tensor(targets)
    def __len__(self):
        return len(self.idx_table)

    def transform_label(self,label):
        if label==self.focusClass:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
    def get_original_label(self,index):
        _,label=self.dataset[self.idx_table[index]]
        return label
    def __getitem__(self, index):
        a=self.idx_table[index]
        datapoint,label=self.dataset[a]
        return datapoint,self.transform_label(label)
        