import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import statistics
plt.rcParams["figure.figsize"] = (30,30)
from matplotlib import colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import itertools
import time
import random
import pickle
from scipy.stats import norm

class imgdataprocess:
    @staticmethod 
    def makeboxes(points,size):
        # Find min,max of both x and y coordinates
        a = np.zeros((2,2))
        a[:,0],a[:,1] = np.min(points, axis=0),np.max(points, axis=0)
        xmin,xmax,ymin,ymax=a[0,0],a[0,1],a[1,0],a[1,1]
        # Calculating number of boxes that can fit in given square
        xinterval,yinterval=xmax-xmin,ymax-ymin
        n_xboxes,n_yboxes=np.floor(xinterval/size),np.floor(yinterval/size)
        # Adjusting xmax and ymax to fit perfect boxes, here you will loose area to the right
        # you can also choose to loose area to left by adjusting xmin and xmax 
        n_xmax,n_ymax=xmin+n_xboxes*size,ymin+n_yboxes*size
        # making x and y points for the square boundaries and making intervals
        xpoints = np.linspace(xmin, n_xmax, num=n_xboxes+1)
        ypoints=np.linspace(ymin,n_ymax, num=n_yboxes+1)
        xintervals=np.array([xpoints[:-1], xpoints[1:]]).transpose()
        yintervals=np.array([ypoints[:-1],ypoints[1:]]).transpose()
        # Making a grid and finally an array of boxes
        grid_boxes=[]
        for f in range(0,len(yintervals)):
            for k in range(0,len(xintervals)):
                grid_boxes.append(np.column_stack([xintervals[k],yintervals[f]]).flatten())
        return grid_boxes
    
    def in_bounding_box(points,box):
        min_x=box[0]
        max_x=box[2]
        min_y=box[1]
        max_y=box[3]
        bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
        bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
        bb_filter = np.logical_and(bound_x, bound_y)
        return len(points[bb_filter])

    def bounding_box_points(points,box):
        min_x=box[0]
        max_x=box[2]
        min_y=box[1]
        max_y=box[3]
        bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
        bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
        bb_filter = np.logical_and(bound_x, bound_y)
        return points[bb_filter]


class getishotspot:
    @staticmethod 
    def check_nb(f,degree,box_index,matrix):
        a,b=box_index[f[0]],box_index[f[1]]
        if a==b:
            matrix[f[0],f[1]]=1
        elif (a[0] ==b[0] and a[2] ==b[2] and a[1] ==b[3] or 
                    a[0] ==b[0] and a[2] ==b[2] and a[3] ==b[1] or 
                    a[1] ==b[1] and a[3] ==b[3] and a[0] ==b[2] or
                    a[1] ==b[1] and a[3] ==b[3] and a[2] ==b[0] or
                    a[3] ==b[1] and a[0] ==b[2] or
                    a[3] ==b[1] and a[2] ==b[0] or
                    a[1] ==b[3] and a[0] ==b[2] or 
                    a[1] ==b[3] and a[2] ==b[0]):
            if degree==1:
                matrix[f[0],f[1]]=1
            elif degree==2:
                matrix[f[0],f[1]]=1
                for g in [(f[1],y) for y in box_index]:
                    c,d=box_index[g[0]],box_index[g[1]]
                    if c==d:
                        matrix[g[0],g[1]]=1
                    elif (c[0] ==d[0] and c[2] ==d[2] and c[1] ==d[3] or 
                    c[0] ==d[0] and c[2] ==d[2] and c[3] ==d[1] or 
                    c[1] ==d[1] and c[3] ==d[3] and c[0] ==d[2] or
                    c[1] ==d[1] and c[3] ==d[3] and c[2] ==d[0] or
                    c[3] ==d[1] and c[0] ==d[2] or
                    c[3] ==d[1] and c[2] ==d[0] or
                    c[1] ==d[3] and c[0] ==d[2] or 
                    c[1] ==d[3] and c[2] ==d[0]):
                        matrix[f[0],g[1]]=1
                    else:
                        matrix[f[0],g[1]]=0
            elif degree==4:
                matrix[f[0],f[1]]=1
                for g in [(f[1],y) for y in box_index]:
                    c,d=box_index[g[0]],box_index[g[1]]
                    if c==d:
                        matrix[g[0],g[1]]=1
                    elif (c[0] ==d[0] and c[2] ==d[2] and c[1] ==d[3] or 
                    c[0] ==d[0] and c[2] ==d[2] and c[3] ==d[1] or 
                    c[1] ==d[1] and c[3] ==d[3] and c[0] ==d[2] or
                    c[1] ==d[1] and c[3] ==d[3] and c[2] ==d[0] or
                    c[3] ==d[1] and c[0] ==d[2] or
                    c[3] ==d[1] and c[2] ==d[0] or
                    c[1] ==d[3] and c[0] ==d[2] or 
                    c[1] ==d[3] and c[2] ==d[0]):
                            matrix[f[0],g[1]]=1
                            for h in [(g[1],y) for y in box_index]:
                                p,q=box_index[h[0]],box_index[h[1]]
                                if p==q:
                                    matrix[h[0],h[1]]=1
                                elif (p[0] ==q[0] and p[2] ==q[2] and p[1] ==q[3] or 
                                p[0] ==q[0] and p[2] ==q[2] and p[3] ==q[1] or 
                                p[1] ==q[1] and p[3] ==q[3] and p[0] ==q[2] or
                                p[1] ==q[1] and p[3] ==q[3] and p[2] ==q[0] or
                                p[3] ==q[1] and p[0] ==q[2] or
                                p[3] ==q[1] and p[2] ==q[0] or
                                p[1] ==q[3] and p[0] ==q[2] or 
                                p[1] ==q[3] and p[2] ==q[0]):
                                    matrix[f[0],h[1]]=1
                                    for i in [(h[1],y) for y in box_index]:
                                        r,s=box_index[i[0]],box_index[i[1]]
                                        if r==s:
                                            matrix[i[0],i[1]]=1
                                        elif (r[0] ==s[0] and r[2] ==s[2] and r[1] ==s[3] or 
                                        r[0] ==s[0] and r[2] ==s[2] and r[3] ==s[1] or 
                                        r[1] ==s[1] and r[3] ==s[3] and r[0] ==s[2] or
                                        r[1] ==s[1] and r[3] ==s[3] and r[2] ==s[0] or
                                        r[3] ==s[1] and r[0] ==s[2] or
                                        r[3] ==s[1] and r[2] ==s[0] or
                                        r[1] ==s[3] and r[0] ==s[2] or 
                                        r[1] ==s[3] and r[2] ==s[0]):
                                            matrix[f[0],i[1]]=1
                                        else:
                                            matrix[f[0],i[1]]=0
                                else:
                                    matrix[f[0],h[1]]=0
                    else:
                        matrix[f[0],g[1]]=0
            
        else:
            matrix[f[0],f[1]]=0

    def detect_hotspot(boxsize,points):
        degree=0
        area=0
        if boxsize==500:
            degree=1
            area=250000
        elif boxsize==200:
            degree=2
            area=40000
        elif boxsize==100:
            degree=4
            area=10000
        boxes=imgdataprocess.makeboxes(points,boxsize)
        boxes_fil={}
        sample= str('Simulated Distribution')
        for f in boxes:
            boxpoints=imgdataprocess.bounding_box_points(points,f)
            if boxpoints.size:
                xmax=boxpoints[:,0].max()
                xmin=boxpoints[:,0].min()
                ymax=boxpoints[:,1].max()
                ymin=boxpoints[:,1].min()
                area_covered=abs((xmax-xmin)*(ymax-ymin))/(area)
                if area_covered > 0.50:
                    boxes_fil[tuple(f)]=imgdataprocess.in_bounding_box(points,f)
        c_dash=statistics.mean(list(boxes_fil.values()))
        box_index={}
        for f in range(0,len(boxes_fil)):
            box_index[f]=list(boxes_fil.keys())[f]
        msize=len(box_index)
        matrix = np.zeros((msize,msize)) 
        masterpermlist=[(x,y) for x in box_index for y in box_index]
        for f in masterpermlist:
            getishotspot.check_nb(f,degree,box_index,matrix)
        
        c_jsum=0
        for f in range(0,msize):
            c_jsum+=list(boxes_fil.values())[f]**2
        S=np.sqrt((c_jsum/msize)-(c_dash**2))
        z_scores={} 
        for k in range(0,msize):
            U=np.sqrt((msize*np.sum(np.square(matrix[k,:]))-np.square(np.sum(matrix[k,:])))/(msize-1))
            wij_cj=0
            for f in range(0,msize):
                wij_cj+=list(boxes_fil.values())[f]*matrix[k,f]
            z_i=(wij_cj-c_dash*np.sum(matrix[k,:]))/S*U
            b_i=box_index[k]
            z_scores[b_i]=z_i
        hotspots=z_scores
        prop_z_5=len({k: v for k, v in hotspots.items() if abs(v) >= 5 }.keys())/len(hotspots)
        prop_z_4=len({k: v for k, v in hotspots.items() if abs(v) >= 4  }.keys())/len(hotspots)
        prop_z_3=len({k: v for k, v in hotspots.items() if abs(v) >= 3  }.keys())/len(hotspots)
        prop_z_2=len({k: v for k, v in hotspots.items() if abs(v) > 2 and abs(v) <=3}.keys())/len(hotspots)
        prop_z_1=len({k: v for k, v in hotspots.items() if abs(v) > 1 and abs(v) <=2}.keys())/len(hotspots)
        prop_z_0=len({k: v for k, v in hotspots.items() if abs(v) >= 0 and abs(v) <=1 }.keys())/len(hotspots)
        result=sample,boxsize,prop_z_0,prop_z_1,prop_z_2,prop_z_3,prop_z_4,prop_z_5
        result_df=pd.DataFrame(list(result)).transpose()
        result_df.columns=['Sample','Boxsize','Prop_Z_0to1','Prop_Z_1to2','Prop_Z_2to3','Prop_Z_gte3','Prop_Z_gte4','Prop_Z_gte5']
        return result_df

    def hotspot_plots(boxsize,points):
        degree=0
        area=0
        if boxsize==500:
            degree=1
            area=250000
        elif boxsize==200:
            degree=2
            area=40000
        elif boxsize==100:
            degree=4
            area=10000
        boxes=imgdataprocess.makeboxes(points,boxsize)
        boxes_fil={}
        sample= str('Simulated Distribution')
        for f in boxes:
            boxpoints=imgdataprocess.bounding_box_points(points,f)
            if boxpoints.size:
                xmax=boxpoints[:,0].max()
                xmin=boxpoints[:,0].min()
                ymax=boxpoints[:,1].max()
                ymin=boxpoints[:,1].min()
                area_covered=abs((xmax-xmin)*(ymax-ymin))/(area)
                if area_covered > 0.50:
                    boxes_fil[tuple(f)]=imgdataprocess.in_bounding_box(points,f)
        c_dash=statistics.mean(list(boxes_fil.values()))
        box_index={}
        for f in range(0,len(boxes_fil)):
            box_index[f]=list(boxes_fil.keys())[f]
        msize=len(box_index)
        matrix = np.zeros((msize,msize)) 
        masterpermlist=[(x,y) for x in box_index for y in box_index]
        for f in masterpermlist:
            getishotspot.check_nb(f,degree,box_index,matrix)
        
        c_jsum=0
        for f in range(0,msize):
            c_jsum+=list(boxes_fil.values())[f]**2
        S=np.sqrt((c_jsum/msize)-(c_dash**2))
        z_scores={} 
        for k in range(0,msize):
            U=np.sqrt((msize*np.sum(np.square(matrix[k,:]))-np.square(np.sum(matrix[k,:])))/(msize-1))
            wij_cj=0
            for f in range(0,msize):
                wij_cj+=list(boxes_fil.values())[f]*matrix[k,f]
            z_i=(wij_cj-c_dash*np.sum(matrix[k,:]))/S*U
            b_i=box_index[k]
            z_scores[b_i]=z_i
        hotspots=z_scores
        prop_z_5=len({k: v for k, v in hotspots.items() if abs(v) > 5 }.keys())/len(hotspots)
        prop_z_4=len({k: v for k, v in hotspots.items() if abs(v) > 4  }.keys())/len(hotspots)
        prop_z_3=len({k: v for k, v in hotspots.items() if abs(v) >= 3  }.keys())/len(hotspots)
        xcors=np.array(list(np.array(list(hotspots.keys()))[:,0])+list(np.array(list(hotspots.keys()))[:,2]))
        ycors=np.array(list(np.array(list(hotspots.keys()))[:,1])+list(np.array(list(hotspots.keys()))[:,3]))
        ##Z-Score Greater than 5 
        z_gt5=np.array(list({k: v for k, v in hotspots.items() if v > 5 }.keys()))
        xcors_0=[]
        ycors_0=[]
        for f in z_gt5:
            xcors_0.append(list(imgdataprocess.bounding_box_points(points,f)[:,0]))
            ycors_0.append(list(imgdataprocess.bounding_box_points(points,f)[:,1]))
        xmerged_0,ymerged_0 = np.array(list(itertools.chain.from_iterable(xcors_0))),np.array(list(itertools.chain.from_iterable(ycors_0)))

        ##Z-Score Greater than 3 , less than 5
        z_gt3=np.array(list({k: v for k, v in hotspots.items() if v > 3 and v <=5 }.keys()))
        xcors_1=[]
        ycors_1=[]
        for f in z_gt3:
            xcors_1.append(list(imgdataprocess.bounding_box_points(points,f)[:,0]))
            ycors_1.append(list(imgdataprocess.bounding_box_points(points,f)[:,1]))
        xmerged_1,ymerged_1 = np.array(list(itertools.chain.from_iterable(xcors_1))),np.array(list(itertools.chain.from_iterable(ycors_1)))

        ##Z-Score Greater than 1 , less than 3
        z_gt1=np.array(list({k: v for k, v in hotspots.items() if v > 1 and v <=3 }.keys()))
        xcors_2=[]
        ycors_2=[]
        for f in z_gt1:
            xcors_2.append(list(imgdataprocess.bounding_box_points(points,f)[:,0]))
            ycors_2.append(list(imgdataprocess.bounding_box_points(points,f)[:,1]))
        xmerged_2,ymerged_2 = np.array(list(itertools.chain.from_iterable(xcors_2))),np.array(list(itertools.chain.from_iterable(ycors_2)))

        ##Z-Score within 1std
        z_1std=np.array(list({k: v for k, v in hotspots.items() if v >= -1 and v <=1 }.keys()))
        xcors_3=[]
        ycors_3=[]
        for f in z_1std:
            xcors_3.append(list(imgdataprocess.bounding_box_points(points,f)[:,0]))
            ycors_3.append(list(imgdataprocess.bounding_box_points(points,f)[:,1]))
        xmerged_3,ymerged_3 = np.array(list(itertools.chain.from_iterable(xcors_3))),np.array(list(itertools.chain.from_iterable(ycors_3)))


        ##Z-Score less than -1 , greater than -3
        z_lt3=np.array(list({k: v for k, v in hotspots.items() if v < -1 and v >=-3 }.keys()))
        xcors_4=[]
        ycors_4=[]
        for f in z_lt3:
            xcors_4.append(list(imgdataprocess.bounding_box_points(points,f)[:,0]))
            ycors_4.append(list(imgdataprocess.bounding_box_points(points,f)[:,1]))
        xmerged_4,ymerged_4 = np.array(list(itertools.chain.from_iterable(xcors_4))),np.array(list(itertools.chain.from_iterable(ycors_4)))


        ##Z-Score Greater than 3 , less than 5
        z_lt5=np.array(list({k: v for k, v in hotspots.items() if v < -3 and v >=-5 }.keys()))
        xcors_5=[]
        ycors_5=[]
        for f in z_lt5:
            xcors_5.append(list(imgdataprocess.bounding_box_points(points,f)[:,0]))
            ycors_5.append(list(imgdataprocess.bounding_box_points(points,f)[:,1]))
        xmerged_5,ymerged_5 = np.array(list(itertools.chain.from_iterable(xcors_5))),np.array(list(itertools.chain.from_iterable(ycors_5)))

        ##Z-Score less than - 5 
        z_lt5=np.array(list({k: v for k, v in hotspots.items() if v < -5 }.keys()))
        xcors_6=[]
        ycors_6=[]
        for f in z_lt5:
            xcors_6.append(list(imgdataprocess.bounding_box_points(points,f)[:,0]))
            ycors_6.append(list(imgdataprocess.bounding_box_points(points,f)[:,1]))
        xmerged_6,ymerged_6 = np.array(list(itertools.chain.from_iterable(xcors_6))),np.array(list(itertools.chain.from_iterable(ycors_6)))
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(xcors)
        ax.set_yticks(ycors)
        ax.grid()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        a=plt.scatter(xmerged_0,ymerged_0,color='red',alpha=0.4)
        b=plt.scatter(xmerged_1,ymerged_1,color='orange',alpha=0.3)
        c=plt.scatter(xmerged_2,ymerged_2,color='yellow',alpha=0.2)
        d=plt.scatter(xmerged_3,ymerged_3,color='grey',alpha=0.1)
        e=plt.scatter(xmerged_4,ymerged_4,color='cyan',alpha=0.2)
        f=plt.scatter(xmerged_5,ymerged_5,color='royalblue',alpha=0.3)
        g=plt.scatter(xmerged_6,ymerged_6,color='blue',alpha=0.4)
        plt.legend((a,b,c,d,e,f,g), ('Z>5','3<Z<5','1<Z<3','-1<Z<1','-3<Z<-1','-3<Z<-5','Z<-5'),scatterpoints=1,loc='best',fontsize=15)
        plt.title("patient: {}, Prop Zgte3 {}, Prop Zgt5 {}".format(sample,prop_z_3,prop_z_5))
        plt.show()
        fig = plt.figure(figsize=(10,10))
        x=np.array(list(hotspots.values()))
        mu,std = np.mean(x),np.std(x)
        n,bins,patches = plt.hist(x, 30, normed=1, facecolor='green', alpha=0.75)
        # add a 'best fit' line
        y = mlab.normpdf( bins, mu, std)
        l = plt.plot(bins, y, 'r--', linewidth=2)
        plt.hist(x, normed=1, bins=10)
        plt.title(r'$\mathrm{Histogram\ of\ hotspots:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, std))
        plt.grid(True)
        plt.show()