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

def find(outputs, labels):
    output_probs = F.softmax(outputs, dim=1)
    x_pred_prob =  (torch.max(output_probs.data, 1)[0][0]) * 100
    _, preds = torch.max(outputs, dim=1) 
    return preds,x_pred_prob

class PneumoniaModelBase(nn.Module):
    

    def validation_step(self, batch):
        images,labels = batch
        out = self(images)                                      # generate predictions
        #loss = F.cross_entropy(out, labels)                     # compute loss
        preds,prob = find(out, labels)                # calculate  preds & probs

    
        return {'preds':preds.detach(),'prob': prob.detach()}

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


data_dir = r"static\chestx"
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

@torch.no_grad()
def test_predict(model, test_loader):
    # perform testing for each batch
    outputs = [model.validation_step(batch) for batch in test_loader]
    preds=(outputs[0]['preds'].item())
    proba=(outputs[0]['prob'].item())
    return preds,proba

UPLOAD=r"static\chestx\test\NORMAL"

def predict(data_dir):
 test_dataset = ImageFolder(data_dir+'/test',
                                 transform=tt.Compose([tt.Resize(255),
                                                       tt.CenterCrop(224),                                                              
                                                       tt.ToTensor()
                                            
                                               ]))

 device='cpu'
 test_dl = DataLoader(test_dataset, batch_size=1)
 test_dl = DeviceDataLoader(test_dl, device)
 preds,proba = test_predict(model, test_dl)
 print(preds)
 print(proba)
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
           if(pred==1):
               return render_template("index.html",prediction="pneumonia",prob=proba,img_loc=image_file.filename)
           if(pred==0):
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