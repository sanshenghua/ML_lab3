from PIL import Image
import numpy as np
import os
import feature as F
from sklearn.model_selection import train_test_split
import ensemble as E
from sklearn.tree import DecisionTreeClassifier
def LoadImages(imgDir,imgFoldName):
    datas = []
    # 读取图片
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    for i in range (imgNum):
        im = Image.open(imgDir+imgFoldName+"/"+imgs[i]) 
        #width,height = im.size
        im = im.resize((24, 24),Image.ANTIALIAS)
        im = im.convert("L") 
        data = im.getdata()
        data = np.array(data)
        #new_data = np.reshape(data,(1,-1))
        #print(data.shape)
        datas.append(data)       
    return datas
if __name__ == "__main__":
    craterDir = "C:/Users/zhancongcong/Desktop/MLdata/ML2017-lab-03-master/datasets/original/"
    foldName1 = "face"
    foldName2 = "nonface"
    X1 = np.array(LoadImages(craterDir,foldName1))#face datas
    X2 = np.array(LoadImages(craterDir,foldName2))#nonface datas
    X = np.concatenate([X1,X2],axis=0)
    print(X.shape)
    features = []
    for i in range(X.shape[0]):	 
        feature = F.NPDFeature(X[i])
        f = feature.extract()  
        features.append(f)
    features = np.array(features)
    print(features.shape)
    y1 = np.ones((500,))#face label
    y2 = -np.ones((500,))#nonface label
    y = np.concatenate([y1,y2],axis=0)
    classifyNum = 10
    eb = E.AdaBoostClassifier(DecisionTreeClassifier, classifyNum)
    #eb.save(features, 'train.spl')
    #features = eb.load('train.spl')
    X_train,X_test,y_train,y_test = train_test_split(features, y, test_size=0.33, random_state=42)
    eb.fit(X_train, y_train)
    eb.predict(X_test)
    eb.report(y_test)
  

