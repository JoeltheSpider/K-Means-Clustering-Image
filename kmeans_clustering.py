import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def rand(x):
    return np.random.randint(x)
    
def dist(cent,pnt):
    return sum([i**2 for i in np.asarray(cent)-np.asarray(pnt)])

def diff(X,Y):
    sum = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            sum+=(X[i][j]-Y[i][j])**2
    sum/=(len(X)*len(X[0]))
    return sum

im = Image.open("image.jpg")
im = im.resize([int(im.size[0]/4),int(im.size[1]/4)])
plt.imshow(im)
plt.title("Original image")
plt.show()

imMap = im.load()
tem = Image.new(im.mode,im.size)
tmap = tem.load()
k=2
r = []
g = []
b = []
for i in range(im.size[0]):
        for j in range(im.size[1]):
            r.append(imMap[i,j][0])
            g.append(imMap[i,j][1])
            b.append(imMap[i,j][2])
sor = sorted(r)
sog = sorted(g)
sob = sorted(b)

while(k<129):
    legends = []
    K = []
    
    #find initial centroids
    for i in range(k):
#        K.append((sor[int((i+1)/(k+1)*len(sor))],sog[int((i+1)/(k+1)*len(sor))],
#                  sob[int((i+1)/(k+1)*len(sor))]))
        K.append((rand(255),rand(255),rand(255)))
        legends.append("Cluster %d"%(i+1))
    
    #K-Means approach to finalize centroids
    print("Approaching KMeans for K ",k)
    prev = [(0,0,0) for i in range(k)]
    while(diff(K,prev)>5):
        cluster = []
        el = [[] for i in range(k)]
        for i in range(im.size[0]):
            for j in range(im.size[1]):
                temp = []
                for t in range(k):
                    temp.append(dist(K[t],imMap[i,j]))
                y = temp.index(min(temp))
                cluster.append(y)
                el[y].append(list(imMap[i,j]))
                tmap[i,j] = K[y]
        prev =  [ i for i in K]
        for i in range (k):
            if el[i]==[]:
                K[i] = (0,0,0)
                continue
            K[i] = []
            K[i].append(int(sum([x[0] for x in el[i]])/len([x[0] for x in el[i]])))
            K[i].append(int(sum([x[1] for x in el[i]])/len([x[1] for x in el[i]])))
            K[i].append(int(sum([x[2] for x in el[i]])/len([x[2] for x in el[i]])))
            K[i] = tuple(K[i])
    
    #plots
    print("Plotting KMeans for K",k)
    f = plt.figure()
    ax = plt.axes(projection='3d')
    img = ax.scatter(r,g,b,c=cluster)   
    ax.set_title("R,G,B spread for %d Ks"%(k))
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    f.colorbar(img)
    plt.show()
    plt.title("%d KMeans image"%(k))
    plt.imshow(tem)
    plt.show()
    tem.save("%d_K_means.png"%(k))
    k = k*2
    print("Done")