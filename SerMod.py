import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from IPython.utils import io
import logging
from NnGen.debug import Debug
from dask import delayed
from dask import compute

class SerMod(nn.Module):
  def __init__(self,layer_spec):
    super(SerMod, self).__init__()
    self.layer_spec=layer_spec
    self.stack=list()
    for i in range(len(layer_spec)-1) :
      lin=nn.Linear(layer_spec[i],layer_spec[i+1])
      self.stack.append(lin)
      if (i<len(layer_spec)-2):
        Relu=nn.ReLU()
        self.stack.append(Relu)
      elif (i==len(layer_spec)-2):
        sig=nn.Sigmoid()
        self.stack.append(sig)
    self.stack=nn.ModuleList(self.stack)
  
  def forward(self, x):
    res=x
    for e in self.stack:
      res=e(res)
    return res

  def serialize_model(self):
    res=torch.tensor([])
    for i in range(0,len(self.stack),2):
      res=torch.cat((res,self.serialize_layer(self.stack[i])))
    return res

  def serialize_layer(self,layer):
    l=layer.state_dict()
    #extract Weights
    w=l['weight'].clone().detach().flatten()
    #extract Biasis
    b=l['bias'].clone().detach().flatten()
    res=torch.cat((w,b))
    return res

  def unserialize_model(self,ser):
    k=0
    for i in range(len(self.layer_spec)-1):
      length=self.layer_spec[i]*self.layer_spec[i+1]+self.layer_spec[i+1]
      l,ser=ser.split([length,len(ser)-length])
      l_weight,l_bias =l.split([self.layer_spec[i]*self.layer_spec[i+1],self.layer_spec[i+1]])
      l_wMat= torch.reshape(l_weight,[self.layer_spec[i+1],self.layer_spec[i]])
      self.stack[k].load_state_dict({'weight':l_wMat,'bias':l_bias}, strict=False)
      k+=2



class light_serial_model(pl.LightningModule):
  def __init__(self,model_vector,model_spec,learning_rate,randomInint=False):
    super().__init__()
    self.model_spec=model_spec
    self.model= SerMod(model_spec)
    if not randomInint:
      self.model.unserialize_model(model_vector)
    self.learning_rate=learning_rate
  def forward(self,x):
    return self.model(x)

  def training_step(self,batch,batch_idx):
    images,labels=batch 
    images=images.reshape(-1,self.model_spec[0])
    output=self.forward(images)  
    Loss=F.mse_loss(output,labels)
    self.log('train_loss',Loss)
    return Loss

  def validation_step(self,batch,batch_idx):
    images,labels=batch 
    images=images.reshape(-1,self.model_spec[0])
    output=self.forward(images)  
    Loss=F.mse_loss(output,labels)
    self.log("val_loss",Loss)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(),lr=self.learning_rate)


class Trainer:
  def __init__(self,batch_size,lr_rate,nb_epoch,dataset,model_spec,random_init=False,num_workers=2,gpus=0):
    self.batch_size=batch_size
    self.lr_rate=lr_rate
    self.nb_epochs=nb_epoch
    self.dataset=dataset
    self.model_spec=model_spec
    self.gpus=gpus
    self.random_init=random_init
    self.num_workers=num_workers
  def create_model(self,data):
    model_vector,_=data
    return light_serial_model(model_vector,self.model_spec,self.lr_rate,randomInint=self.random_init)
  
  def create_dataLoader(self,data):
    _,clas=data
    return DataLoader(dataset=self.dataset.get_subDatast(clas),batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True)
  
  def train(self,data):
    model,subdataloader=data
    #with io.capture_output() as captured:
      #logging.getLogger("lightning").setLevel(logging.ERROR)
    model_trainer=pl.Trainer(callbacks=[EarlyStopping(monitor="train_loss",min_delta=0.0, mode="min")],max_epochs=self.nb_epochs,enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=True,logger=False,gpus=self.gpus)
    model_trainer.fit(model=model,train_dataloaders=subdataloader)
    
  def train_models(self,data):
    models=[(self.create_model(m),self.create_dataLoader(m)) for m in data]
    list_of_delayed_functions = []
    for d in models:
      list_of_delayed_functions.append(delayed(self.train)(d))
    compute(list_of_delayed_functions, num_workers=self.num_workers)
    return self.serialize_models(models) 

  def serialize_models(self,models):
    res=None
    for model,_ in models:
      serial_model=model.model.serialize_model()
      if res==None:
        res=serial_model.reshape(-1,len(serial_model))
      else:
        res=torch.cat([res,serial_model.reshape(-1,len(serial_model))])
    return res



class Trainer2 :
    def __init__(self,batch_size,lr_rate,nb_epoch,input_size,dataset,num_workers=2,zeroTrain=False):
        self.batch_size=batch_size
        self.lr_rate=lr_rate
        self.nb_epoch=nb_epoch
        self.input_size=input_size
        self.debug=Debug()
        self.zeroTrain=zeroTrain
        self.dataset=dataset
        self.num_workers=num_workers
    def train(self,data,modelSpec):
      
        models=[]
        for i,d in enumerate(data):
            vmodel,clas=d
            model=SerMod(modelSpec)
            if not self.zeroTrain:
              model.unserialize_model(vmodel)
            dataloader=DataLoader(self.dataset.get_subDatast(clas),num_workers=self.num_workers,batch_size=self.batch_size,shuffle=True,pin_memory=True)
            models.append({'model':model,'loader':dataloader,'idx':i})
        #pool = multiprocessing.Pool()
        #pool = multiprocessing.Pool(processes=10)
        #threads=[]
        #for model in models:
        #  th=threading.Thread(target=self.train_model,args=(model,))
        #  threads.append(th)
        #  th.start()

        #for thread in threads:
        #  thread.join()
          #self.train_model(model)

        list_of_delayed_functions = []
        for d in models:
          list_of_delayed_functions.append(delayed(self.train_model)(d))
        models_loss=compute(list_of_delayed_functions)
        
        #models_loss=map(self.train_model,models)
        #print(model.serialize_model())
        res=None
        for model in models:
          serial_model=model['model'].serialize_model()
          if res==None:
            res=serial_model.reshape(-1,len(serial_model))
          else:
            res=torch.cat([res,serial_model.reshape(-1,len(serial_model))])
        return res


    def train_model(self ,data):
      loss=nn.MSELoss()
      optimizer=torch.optim.Adam(data['model'].parameters(),lr=self.lr_rate)
      for epoch in range(self.nb_epoch):
        for i,(images,labels) in enumerate(data['loader']):
          images=images.reshape(-1,self.input_size)
          output=data['model'](images)  
          Loss=loss(output,labels)
          optimizer.zero_grad()
          Loss.backward()
          optimizer.step()
          self.debug.writeUpdate('  model:'+str(data['idx'])+'  epoch:'+str(epoch)+'  Loss:'+str(Loss.item()))
      self.debug.write('\n')
      return Loss.item()

