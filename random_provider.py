import numpy as np


class provider():

    def __init__(self,raw,aff,gt,shape):
        self.raw=raw
        self.aff=aff
        self.gt=gt
        self.shape=shape

        self.num_rots=self.number_of_rotations()

        self.raw_rotations=np.zeros(self.num_rots+1,dtype=object)
        self.aff_rotations=np.zeros(self.num_rots+1,dtype=object)
        self.gt_rotations=np.zeros(self.num_rots+1,dtype=object)

        self.raw_rotations[0]=np.asarray(raw,dtype=object)
        self.aff_rotations[0]=np.asarray(aff,dtype=object)
        self.gt_rotations[0]=np.asarray(gt,dtype=object)

        self.raw_rotations[1:],self.aff_rotations[1:],self.gt_rotations[1:]=self.create_rotations()

    def number_of_rotations(self):
        n=0
        for i in range(0,3):
            for j in range(i,3):
                if(i==j):
                    continue
                if(self.shape[i]==self.shape[j]):
                    n+=1
        return n

    def create_rotations(self):
        #swap all dim permuations that are equal and populate list with them

        raw_arr=np.zeros(self.num_rots,dtype=object)
        aff_arr=np.zeros(self.num_rots,dtype=object)
        gt_arr=np.zeros(self.num_rots,dtype=object)

        n=0
        for i in range(0,3):
            for j in range(i,3):
                if(i==j):
                    continue
                if(self.shape[i]==self.shape[j]):
                    raw_arr[n]=np.asarray(np.swapaxes(self.raw,i,j),dtype=object)
                    aff_arr[n]=np.asarray(np.swapaxes(self.aff,i,j),dtype=object)
                    gt_arr[n]=np.asarray(np.swapaxes(self.gt,i,j),dtype=object)
                    n+=1
        return raw_arr,aff_arr,gt_arr

    def random_provider_affin(self):

        x = int(np.random.random() * (np.shape(self.raw)[0] - self.shape[0]))
        y = int(np.random.random() * (np.shape(self.raw)[1] - self.shape[1]))
        z = int(np.random.random() * (np.shape(self.raw)[2] - self.shape[2]))
        raw_out = np.zeros((1, 1, self.shape[0], self.shape[1], self.shape[2]))
        aff_out = np.zeros((1, 3, self.shape[0], self.shape[1], self.shape[2]))
        raw_out[0] = self.raw[x:x + self.shape[0], y:y + self.shape[1], z:z + self.shape[2]]
        aff_out[0] = self.affin[:, x:x + self.shape[0], y:y + self.shape[1], z:z + self.shape[2]]
        raw_out = np.einsum("bczxy->bzxyc", raw_out)
        aff_out = np.einsum("bczxy->bzxyc", aff_out)

        self.rotate()

        return raw_out, aff_out

    def random_provider_affin_gt(self):
        affin=1



        x=int(np.random.random()*(np.shape(self.raw)[0]-self.shape[0]))
        y=int(np.random.random()*(np.shape(self.raw)[1]-self.shape[1]))
        z=int(np.random.random()*(np.shape(self.raw)[2]-self.shape[2]))

        raw_out=np.zeros((1,1,self.shape[0],self.shape[1],self.shape[2]))
        aff_out=np.zeros((1,3,self.shape[0],self.shape[1],self.shape[2]))
        gt_out = np.zeros((1, 1, self.shape[0], self.shape[1], self.shape[2]))

        raw_out[0]=self.raw[x:x+self.shape[0],y:y+self.shape[1],z:z+self.shape[2]]
        aff_out[0]=self.affin[:,x:x+self.shape[0],y:y+self.shape[1],z:z+self.shape[2]]
        gt_out[0]=self.gt[x:x+self.shape[0],y:y+self.shape[1],z:z+self.shape[2]]

        raw_out=np.einsum("bczxy->bzxyc",raw_out)
        aff_out = np.einsum("bczxy->bzxyc", aff_out)
        gt_out = np.einsum("bczxy->bzxyc", self.gt)

        #if (np.random.random() > .5):
        ##    raw_out = np.einsum("bczxy->bczyx", raw_out)
         #  aff_out = np.einsum("bczxy->bczyx", aff_out)

        self.rotate()

        return raw_out,aff_out, gt_out

    def random_provider_loss_info(self):

        x = int(np.random.random() * (np.shape(self.raw)[0] - self.shape[0]))
        y = int(np.random.random() * (np.shape(self.raw)[1] - self.shape[1]))
        z = int(np.random.random() * (np.shape(self.raw)[2] - self.shape[2]))

        raw_out = np.zeros((1, 1, self.shape[0], self.shape[1], self.shape[2]))
        loss_info_out = np.zeros((1, 4, self.shape[0], self.shape[1], self.shape[2]))

        raw_out[0] = self.raw[x:x + self.shape[0], y:y + self.shape[1], z:z + self.shape[2]]
        loss_info_out[0,0:3] = self.aff[:, x:x + self.shape[0], y:y + self.shape[1], z:z + self.shape[2]]
        loss_info_out[0,3]=self.gt[x:x + self.shape[0], y:y + self.shape[1], z:z + self.shape[2]]


        raw_out = np.einsum("bczxy->bzxyc", raw_out)
        loss_info_out = np.einsum("bczxy->bzxyc", loss_info_out)

        self.rotate()

        return raw_out, loss_info_out

    def random_provider_raw(self):

        x = int(np.random.random() * (np.shape(self.raw)[0] - self.shape[0]))
        y = int(np.random.random() * (np.shape(self.raw)[1] - self.shape[1]))
        z = int(np.random.random() * (np.shape(self.raw)[2] - self.shape[2]))

        self.rotate()

        return self.raw[x:x + self.shape[0], y:y + self.shape[1], z:z + self.shape[2]]

    def rotate(self):
        num=int(np.random.random()*self.num_rots)

        self.raw=self.raw_rotations[num]
        self.aff=self.aff_rotations[num]
        self.gt=self.gt_rotations[num]

    def reset_rotations(self):
        self.raw = self.raw_rotations[0]
        self.aff = self.aff_rotations[0]
        self.gt = self.gt_rotations[0]










