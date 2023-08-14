def help():
    print("깃허브 레포지트리를 참고해주세요!\n[Dev_Github](Layla-Focalors):https://github.com/layla-focalors\n기본 사용 메소드 : import layla-focalors.cyclegan as lfc")
    return None

def loadcuda():
    import platform
    print("해당 함수는 현재 지원하지 않습니다. 업데이트를 기다려주세요!")

def loadmodel_cycle_test(preset, inputpath, outputpath, epoch):
    import os 
    import torch 
    import torchvision 
    import warnings
    from torch.utils.data import DataLoader
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 로컬변수
    def get_data_loader(image_type, image_dir, image_size=128, batch_size=16, num_workers=0):
        """Returns training and test data loaders for a given image type, size and directory"""
        # resize and normalize the images
        transform = transforms.Compose([transforms.Resize(image_size), # resize to 128x128
                                        transforms.ToTensor()])

        # get training and test directories
        image_path = os.path.join(image_dir, image_type)
        train_path = os.path.join(image_path, 'train')
        test_path = os.path.join(image_path, 'test')

        # define datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_path, transform)
        test_dataset = datasets.ImageFolder(test_path, transform)

        # create and return DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader
    dataloader_X, test_dataloader_X = get_data_loader(image_type='summer')
    dataloader_Y, test_dataloader_y = get_data_loader(image_type='winter')
    
    # prs
    dataiter = iter(dataloader_Y)
    images, _ = dataiter.next()
    
    # preprocess
    
    img = images[0]
    # print(f"Min: {img.min()}")
    # print(f"Max: {img.max()}")
    
    def scale(x, feature_range=(-1, 1)):
        ''' Scale takes in an image x and returns that image, scaled
           with a feature_range of pixel values from -1 to 1. 
           This function assumes that the input x is already scaled from 0-1.'''
        
        # assume x is scaled to (0, 1)
        # scale to feature_range and return scaled x
        min, max = feature_range
        x = x * (max - min) + min
        return x
    
    scaled_img = scale(img)
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        """Creates a convolutional layer, with optional batch normalization.
        """
        layers = []
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        layers.append(conv_layer)

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    class Discriminator(nn.Module):
        def __init__(self, conv_dim=64):
            super(Discriminator, self).__init__()
            
            self.conv1 = conv(3, conv_dim * 2, 4)
            self.conv2 = conv(conv_dim, conv_dim * 2, 4)
            self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
            self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
            self.conv5 = conv(conv_dim * 8, 1, 4, stride=1, batch_norm=False)
        
        def forward(self, x):
            x = F.leaky_relu(self.conv1(x), 0.2)
            x = F.leaky_relu(self.conv2(x), 0.2)
            x = F.leaky_relu(self.conv3(x), 0.2)
            x = F.leaky_relu(self.conv4(x), 0.2)
            x = self.conv5(x)
            return x
        
    class ResidualBlock(nn.Module):
        def __init__(self, conv_dim):
            super(ResidualBlock, self).__init__()
            # conv_dim = number of inputs
        
            self.conv1 = conv(conv_dim, conv_dim, 3, 1, 1, batch_norm=True)
            self.conv2 = conv(conv_dim, conv_dim, 3, 1, 1, batch_norm=True)
            
        def forward(self, x):
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out)) + x
            return out
    
    class CycleGenerator(nn.Module):
        def __init__(self, conv_dim=64, n_res_block=6):
            super(CycleGenerator, self).__init__()
            
            # 1. Define the encoder part of the generator
            self.conv1 = conv(3, conv_dim, 4)
            self.conv2 = conv(conv_dim, conv_dim * 2, 4)
            self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
            
            # 2. Define the resnet part of the generator
            res_layers = []
            for layer in range(n_res_block):
                res_layers.append(ResidualBlock(conv_dim * 4))
            self.res_blocks = nn.Sequential(*res_layers)
            
            # 3. Define the decoder part of the generator
            self.deconv1 = conv(conv_dim * 4, conv_dim * 2, 4)
            self.deconv2 = conv(conv_dim * 2, conv_dim, 4)
            self.deconv3 = conv(conv_dim, 3, 4, batch_norm=False)
            
        def forward(self, x):
            """Given an image x, returns a transformed image."""
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
            out = F.relu(self.conv3(out))
            
            out = self.res_blocks(out)
            
            out = F.relu(self.deconv1(out))
            out = F.relu(self.deconv2(out))
            out = F.tanh(self.deconv3(out))
            return out
        
    def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        """Creates a transposed-convolutional layer, with optional batch normalization.
            """
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
        """Builds the generators and discriminators."""
        
        # Instantiate generators
        G_XtoY = CycleGenerator(conv_dim=g_conv_dim, n_res_block=n_res_blocks)
        G_YtoX = CycleGenerator(conv_dim=g_conv_dim, n_res_block=n_res_blocks)
        # Instantiate discriminators
        D_X = Discriminator(conv_dim=d_conv_dim)
        D_Y = Discriminator(conv_dim=d_conv_dim)

        # move models to GPU, if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            G_XtoY.to(device)
            G_YtoX.to(device)
            D_X.to(device) 
            D_Y.to(device)
            print('Models moved to GPU.')
        else:
            print('Only CPU available.')
        
        return G_XtoY, G_YtoX, D_X, D_Y
    
    G_XtoY, G_YtoX, D_X, D_Y = create_model()
    
    def real_mse_loss(D_out):
        # 얼마나 판별자의 출력이 진짜에 가까운지
        # how close is the produced output from being "real"?
        return torch.mean((D_out-1)**2)
    
    def fake_mse_loss(D_out):
        # 얼마나 판별자의 출력이 가짜에 가까운지
        # how close is the produced output from being "fake"?
        return torch.mean(D_out**2)
    
    def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
        # 실제 영상 오차와 가짜 오차와의 평균절대오차 반환 , 가중치를 부여하는 lambda_weight 포함
        # calculate reconstruction loss 
        # as absolute value difference between the real and reconstructed images
        recon_loss = torch.mean(torch.abs(real_im - reconstructed_im))
        # return weighted loss
        return lambda_weight*recon_loss
    
    import torch.optim as optim
    # hyperparams for Adam optimizers
    lr=0.0002
    beta1=0.5
    beta2=0.999 # default value
    
    # Create optimizers for the generators and discriminators
    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

    # by_gpu passport
    # d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters
    
    g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
    d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
    d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])
    
    # from helpers2 import save_samples, checkpoint
    def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_epochs=1000):
        point_every = 100
        # keep track of losses over time
        losses = []
        test_iter_X = iter(test_dataloader_X)
        test_iter_Y = iter(test_dataloader_Y)
        
        # Get some fixed data from domains X and Y for sampling. These are images that are held
        fixed_X = test_iter_X.next()[0]
        fixed_Y = test_iter_Y.next()[0]
        fixed_X = scale(fixed_X)
        fixed_Y = scale(fixed_Y)
        
        # batches per epoch
        iter_X = iter(dataloader_X)
        iter_Y = iter(dataloader_Y)
        batches_per_epoch = min(len(iter_X), len(iter_Y))
        
        for epoch in range(1, n_epochs+1):
            if epoch % batches_per_epoch == 0:
                iter_X = iter(dataloader_X)
                iter_Y = iter(dataloader_Y)
                
        images_X, _ = iter_X.next()
        images_X = scale(images_X) # make sure to scale to a range -1 to 1
        
        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)
    
        d_x_optimizer.zero_grad()
        out_x = D_X(images_X)
        D_X_loss = real_mse_loss(out_x)
        
        fake_X = G_YtoX(images_Y)
        D_X_fake_loss = fake_mse_loss(fake_X)
        
        d_x_loss = D_X_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()
        
        d_y_optimizer.zero_grad()
        out_y = D_Y(images_Y)
        D_Y_real_loss = real_mse_loss(out_y)
        
        fake_Y = G_XtoY(images_X)
        D_Y_fake_loss = fake_mse_loss(fake_Y)
        
        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()
        
        g_optimizer.zero_grad()
        fake_X = G_YtoX(images_Y)
        out_X = D_X(fake_X)
        g_YtoX_loss = real_mse_loss(out_X)
        recon_Y = G_XtoY(fake_X)
        recon_Y = cycle_consistency_loss(images_Y, recon_Y, lambda_weight=10)
        fake_Y = G_XtoY(images_X)
        out_Y = D_Y(fake_Y)
        g_XtoY_loss = real_mse_loss(out_Y)
        recon_X = G_YtoX(fake_Y)
        recon_X = cycle_consistency_loss(images_X, recon_X, lambda_weight=10)
        g_total_loss = g_YtoX_loss + g_XtoY_loss + recon_X + recon_Y
        g_total_loss.backward()
        g_optimizer.step()
        
        print_every = 10
        
        if epoch % print_every == 0:
            losses.append((d_x_loss.item(), d_y_loss.item(), g_YtoX_loss.item(), g_XtoY_loss.item()))
            
        sample_every = 100
        if epoch % sample_every == 0:
            G_YtoX.eval()
            G_XtoY.eval()
            # save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16)
            G_YtoX.train()
            G_XtoY.train()
            
        return losses        

def mloadmodel(model_path, inputpath, outputpath, epoch):
    pass

def editpreset(preset_name:str):
    import os
    import sqlite3
    os.chdir("preset")
    conn = sqlite3.connect("preset.db")
    cur = conn.cursor()
    settings = ["model_path", "webui"]
    for i in range(len(settings)):
        if i == 0:
            print("모델의 경로를 지정하시겠습니까?")
            print("이미 존재하는 모델을 사용하는 경우 설정해주세요! ( 기본값 : Notdefined )")
            ses = input("모델의 경로를 지정할까요? (Y/N) : ")
            if ses == "Y" or ses == "y":
                model_path = input("모델의 경로를 입력해주세요\n")
                print("모델의 경로가 지정되었습니다. 수정하시려면 edit model을 사용해주세요!")
            elif ses == "N" or ses == "n":
                print("모델의 경로를 지정하지 않습니다.")
                model_path = "Undefined"
            else:
                print("잘못된 입력입니다. 모델의 경로를 지정하지 않습니다.!")
                model_path = "Undefined"
        elif i == 1:
            webui = input("웹UI를 사용하시겠습니까? ( 기본값 : Y ) : ")
            if webui == "Y" or webui == "y":
                webui_value = 1
            elif webui == "N" or webui == "n":
                webui_value = 0
            else:
                print("잘못된 입력입니다. 기본값이 적용됩니다.")
    print("-------------------------------------------------")
    print("설정을 완료했습니다. 다음과 같은 설정이 적용됩니다.")
    setting_dat = [model_path, webui_value]
    for i in range(len(settings)):
        print(f"Option {settings[i]} : {setting_dat[i]}")
    print("-------------------------------------------------")
    cur.execute(f"INSERT INTO preset(preset, model_path, webui) VALUES (?,?,?)",(preset_name, model_path, webui_value))
    conn.close()
    return None

# def web_edit_preset():
    # from fastapi import FastAPI
    # from fastapi.responses import StreamingResponse
    # from fastapi import Request
    # from fastapi.responses import HTMLResponse
    # from fastapi.templating import Jinja2Templates
    # from fastapi.staticfiles import StaticFiles
    # import pymysql
    # from pydantic import BaseModel
    # templates = Jinja2Templates(directory="templates")
    # app = FastAPI(docs_url="/documentation", redoc_url=None)
    # @app.get("/")
    # async def home(request: Request):
    #     return templates.TemplateResponse("index.html",{"request":request})
    # print("webui를 실행했습니다.")
    # 현재 지원하지 않는 기능입니다. / edit preset을 당분간 이용해주세요!

def mkpreset(preset_name:str):
    import os 
    import sqlite3
    # package load
    if os.path.exists("preset"):
        os.chdir("preset")
        conn = sqlite3.connect("preset.db")
        cur = conn.cursor()
    else:
        os.system("mkdir preset")
        os.chdir("preset")
        conn = sqlite3.connect("preset.db")
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS preset (preset TEXT, model_path TEXT, webui INTEGER)")
    # if database == True, Bypass command, else, make database
    print("프리셋 설정을 진행해주세요!")
    settings = ["model_path", "webui"]
    for i in range(len(settings)):
        if i == 0:
            print("모델의 경로를 지정하시겠습니까?")
            print("이미 존재하는 모델을 사용하는 경우 설정해주세요! ( 기본값 : Notdefined )")
            ses = input("모델의 경로를 지정할까요? (Y/N) : ")
            if ses == "Y" or ses == "y":
                model_path = input("모델의 경로를 입력해주세요\n")
                print("모델의 경로가 지정되었습니다. 수정하시려면 edit model을 사용해주세요!")
            elif ses == "N" or ses == "n":
                print("모델의 경로를 지정하지 않습니다.")
                model_path = "Undefined"
            else:
                print("잘못된 입력입니다. 모델의 경로를 지정하지 않습니다.!")
                model_path = "Undefined"
        elif i == 1:
            webui = input("웹UI를 사용하시겠습니까? ( 기본값 : Y ) : ")
            if webui == "Y" or webui == "y":
                webui_value = 1
            elif webui == "N" or webui == "n":
                webui_value = 0
            else:
                print("잘못된 입력입니다. 기본값이 적용됩니다.")
    print("-------------------------------------------------")
    print("설정을 완료했습니다. 다음과 같은 설정이 적용됩니다.")
    setting_dat = [model_path, webui_value]
    for i in range(len(settings)):
        print(f"Option {settings[i]} : {setting_dat[i]}")
    print("-------------------------------------------------")
    cur.execute(f"INSERT INTO preset(preset, model_path, webui) VALUES (?,?,?)",(preset_name, model_path, webui_value))
    conn.close()
    # TestCode Init
    print(f"프리셋 {preset_name}가 생성되었습니다!")
    return None
    
def remove_all_preset():
    sas = input("[경고] 해당 작업은 등록된 모든 프리셋을 삭제합니다. 진행하시겠습니까? (Y/N) : ")
    if sas == "Y" or sas == "y":
        saa = input("해당 작업은 돌이킬 수 없습니다. 정말 진행하시겠습니까? (Y/N) : ")
        if saa == "Y" or saa == "y":
            print("모든 프리셋을 삭제합니다.")
            import os
            import shutil
            print(os.getcwd())
            try:
                shutil.rmtree("preset")
            except OSError as e:
                print(f"오류가 발생했습니다. 오류 : {e}")
            print("프리셋을 삭제했습니다.")
        elif saa == "N" or saa == "n":
            print("작업을 취소했습니다.")
        else:
            print("잘못된 입력입니다. 작업을 취소했습니다.")
    elif sas == "N" or sas == "n":
        print("작업을 취소했습니다.")
        return None
    else:
        print("잘못된 입력입니다. 작업을 취소했습니다.")
        return None
    return None
    
    