from flask import Flask
from flask import render_template
import os
from flask  import request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from torch.utils.data import  Dataset
import jinja2
from PIL import Image
import io
import requests
app=Flask(__name__)
device='cpu'

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device) # yield will stop here, perform other steps, and the resumes to the next loop/batch

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def accuracy(outputs, labels):
    output_probs = F.softmax(outputs, dim=1)
    x_pred_prob =  (torch.max(output_probs.data, 1)[0][0]) * 100
    _, preds = torch.max(outputs, dim=1) 
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds,x_pred_prob

def F1_score(outputs, labels):
    _, preds = torch.max(outputs, dim=1) 
    
    # precision, recall, and F1
    cm  = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))
    
    return precision,recall,f1,preds
class PneumoniaModelBase(nn.Module):
    
    # this is for loading the batch of train image and outputting its loss, accuracy 
    # & predictions
    def training_step(self, batch, weight):
        images,labels = batch
        out = self(images)                                      # generate predictions
        loss = F.cross_entropy(out, labels, weight=weight)      # weighted compute loss
        acc,preds = accuracy(out, labels)                       # calculate accuracy
        
        return {'train_loss': loss, 'train_acc':acc}
       
    # this is for computing the train average loss and acc for each epoch
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]       # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['train_acc'] for x in outputs]          # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
        
        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}
         # this is for loading the batch of val/test image and outputting its loss, accuracy, 
    # predictions & labels
    def validation_step(self, batch):
        images,labels = batch
        out = self(images)                                      # generate predictions
        loss = F.cross_entropy(out, labels)                     # compute loss
        acc,preds,prob = accuracy(out, labels)                # calculate acc & get preds 
        return {'val_loss': loss.detach(), 'val_acc':acc.detach(), 
               'preds':preds.detach(), 'labels':labels.detach(),'prob': prob.detach()}
    # detach extracts only the needed number, or other numbers will crowd memory
    
    # this is for computing the validation average loss and acc for each epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]         # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['val_acc'] for x in outputs]            # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
        
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        # this is for printing out the results after each epoch
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.
              format(epoch+1, train_result['train_loss'], train_result['train_acc'],
                     val_result['val_loss'], val_result['val_acc']))
    
    # this is for using on the test set, it outputs the average loss and acc, 
    # and outputs the predictions
    def test_prediction(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
        batch_prob = [x['prob'] for x in outputs]
        epoch_prob = torch.stack(batch_prob).mean()  
        # combine predictions
        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()] 
        # combine labels
        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]  
        
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(),
                'test_preds': batch_preds, 'test_labels': batch_labels,'proba':epoch_prob.item()}
resnet50 = models.resnet50(pretrained=True)
#resnet50

class PneumoniaResnet(PneumoniaModelBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Freeze training for all layers before classifier
        for param in self.network.fc.parameters():
            param.require_grad = False  
        num_features = self.network.fc.in_features # get number of in features of last layer
        self.network.fc = nn.Linear(num_features, 2) # replace model classifier
    
    def forward(self, xb):
        return self.network(xb)
    
#     def freeze(self):
#         # To freeze the residual layers
#         for param in self.network.parameters():
#             param.require_grad = False
#         for param in self.network.fc.parameters():
#             param.require_grad = True
    
#     def unfreeze(self):
#         # Unfreeze all layers
#         for param in self.network.parameters():
#             param.require_grad = True



data_dir = r"static\chestx"
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

#model = load_checkpoint(r'C:\Users\ASUS !\Downloads\PneumoniaResnet.pth')


@torch.no_grad()
def test_predict(model, test_loader):
    # perform testing for each batch
    outputs = [model.validation_step(batch) for batch in test_loader] 
    results = model.test_prediction(outputs)     


    #print('test_loss: {:.4f}, test_acc: {:.4f}\nPrediction: {}'
    #      .format(results['test_loss'], results['test_acc'],results['test_preds']))
    #print(results['test_labels'])
    return results['test_preds'], results['test_labels'],results['proba']


my_transform=tt.Compose([tt.Resize(255),
                         tt.CenterCrop(224),                                                              
                         tt.ToTensor()
                                                 #tt.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 #             std=[0.229, 0.224, 0.225],
                                                 #             inplace=True)
                         ])

UPLOAD=r"static\chestx\test\NORMAL"
# Plot Accuracy and Loss 
def predict(data_dir):
 test_dataset = ImageFolder(data_dir+'/test',
                                 transform=tt.Compose([tt.Resize(255),
                                                       tt.CenterCrop(224),                                                              
                                                       tt.ToTensor()
                                                 #tt.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 #             std=[0.229, 0.224, 0.225],
                                                 #             inplace=True)
                                               ]))

# Evaluate test set
 device='cpu'
 test_dl = DataLoader(test_dataset, batch_size=1)
 test_dl = DeviceDataLoader(test_dl, device)
 preds,labels,proba = test_predict(model, test_dl)
 preds,labels,proba= test_predict(model, test_dl)
 return preds,proba
@app.route("/index", methods=["GET" ,"POST"])
def upload_predict():
    mypath = r"static\chestx\test" #Enter your path here
    for root, dirs, files in os.walk(mypath):
            for file in files:
                os.remove(os.path.join(root, file))
    if request.method == "POST":
       image_file=request.files["image"]
       if image_file:
           image_location=os.path.join(
            UPLOAD,
               image_file.filename
           )
           image_file.save(image_location)
           pred,proba=predict(r"static\chestx")
           #os.remove(image_location)
           if(pred==[1]):
               return render_template("index.html",prediction="pneumonia",prob=proba,img_loc=image_file.filename)
           if(pred==[0]):
               return render_template("index.html",prediction="Normal",prob=proba,img_loc=image_file.filename)

    return render_template("index.html",prediction=" ")
@app.route("/symptom")
def symptom():
    return render_template('symptom.html')
@app.route("/types")
def types():
    return render_template('types.html')
@app.route("/diagnosis")
def diagnosis():
    return render_template('diagnosis.html')
@app.route("/covid", methods=["GET" ,"POST"])
def covid():
    dat=requests.get("https://disease.sh/v2/countries")
    world=requests.get("https://disease.sh/v3/covid-19/all")
    dat_dict=dat.json()
    world_dict=world.json()
    if request.method == "POST":
       r=request.form['country']
       a_string =r
       print(a_string)
      
       num = ''.join(filter(lambda i: i.isdigit(),r))
       print(num)
       print(type(num))
       number=int(num)
       print(type(number))
       print(number)
       return render_template("covid.html",data=dat_dict,p=number)
    return render_template("covidworld.html",world=world_dict)
if __name__ == "__main__":
 model = load_checkpoint('./PneumoniaResnet.pth')
 app.run(debug=True)