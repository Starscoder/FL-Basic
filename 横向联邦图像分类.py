import numpy as np
import torch
import copy

from torchvision import datasets, transforms

def get_dataset(dir, name):

    if name=='mnist':
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
        
    elif name=='cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                        transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
        
    
    return train_dataset, eval_dataset


class Server(object):
    
    def __init__(self, conf, eval_dataset):
    
        self.conf = conf 
        
        self.global_model = get_model(self.conf["model_name"]) 
        
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
        
    
    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
                
    def model_eval(self):
        self.global_model.eval()
        
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch 
            dataset_size += data.size()[0]
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                
            
            output = self.global_model(data)
            
            total_loss += torch.nn.functional.cross_entropy(output, target,
                                              reduction='sum').item() # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l


class Client(object):

    def __init__(self, conf, model, train_dataset, id = -1):
        
        self.conf = conf
        
        self.local_model = get_model(self.conf["model_name"]) 
        
        self.client_id = id
        
        self.train_dataset = train_dataset
        
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
                                    
        
    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
    
        #print(id(model))
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])
        #print(id(self.local_model))
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
            
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
            
                optimizer.step()
            print("Epoch %d done." % e) 
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
            #print(diff[name])
            
        return diff

from torchvision import models

def get_model(name="vgg16", pretrained=True):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)  
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)       
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":       
        model = models.googlenet(pretrained=pretrained)
        
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model 



import argparse, json
import datetime
import os
import logging
import torch, random


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()
    

    with open(args.conf, 'r') as f:
        conf = json.load(f) 
    
    
    train_datasets, eval_datasets = get_dataset("data", conf["type"])
    
    server = Server(conf, eval_datasets)
    clients = []
    
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))
        
    print("\n\n")
    for e in range(conf["global_epochs"]):
    
        candidates = random.sample(clients, conf["k"])
        
        weight_accumulator = {}
        
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        
        for c in candidates:
            diff = c.local_train(server.global_model)
            
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
                
        
        server.model_aggregate(weight_accumulator)
        
        acc, loss = server.model_eval()
        
        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
