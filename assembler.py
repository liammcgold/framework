import numpy as np
import blender as blend
import math
import tifffile as tif


class assembler(object):

    def __init__(self,raw,overlap,shape,function,blend_fac):
        self.raw=raw
        self.raw_shape = np.shape(raw)
        self.counter = 0
        self.stage = 0
        self.shape = shape
        self.full = False
        self.function = function
        self.raw_block = None
        self.aff_block = None
        self.blend_fac=blend_fac
        self.__initialize_overlap(overlap)

        self.overlap_increment = self.calc_overlap_inc()

        self.num_blocks,self.padding=self.calc_num_blocks()

        self.total_blocks=self.calc_total_blocks()

        self.aff_graph=self.generate_empty_aff_graph()

        self.dict=self.on_edge()

        self.current_loc=self.get_raw_coordinates()

        self.buffer0=np.zeros((3,np.shape(self.aff_graph)[1],self.shape[1],self.shape[2]))

    def __initialize_overlap(self,overlap):

        if(overlap==0):
            self.overlap=0
            return

        print(self.shape)

        a=math.gcd(self.shape[0],self.shape[1])
        b=math.gcd(self.shape[1],self.shape[2])
        c=math.gcd(self.shape[0],self.shape[2])

        gcd = np.min([a,b,c])

        if math.ceil(overlap*gcd)==gcd:
            overlap=math.floor(overlap*gcd)/gcd
        else:
            overlap=math.ceil(overlap*gcd)/gcd

        self.overlap=overlap

    def calc_total_blocks(self):
        total_blocks = self.num_blocks[0] * self.num_blocks[1] * self.num_blocks[2]
        return total_blocks

    def generate_empty_aff_graph(self):
        aff_graph = np.zeros((3,
                              int(self.raw_shape[0]+ self.padding[0]*self.shape[0]),
                              int(self.raw_shape[1]+ self.padding[1]*self.shape[1]),
                              int(self.raw_shape[2]+ self.padding[2]*self.shape[2])
                              ))
        return aff_graph

    def calc_overlap_inc(self):
        if self.overlap==0:
            return self.shape

        overlap_increment = [self.shape[0] - self.overlap * self.shape[0], self.shape[1] - self.overlap * self.shape[1],
                             self.shape[2] - self.overlap * self.shape[2]]
        return overlap_increment

    def feed(self):


        assert self.stage==0, "CALLED FEED DURRING APPEND STAGE"

        if(self.is_full()):
            print("NO MORE BLOCKS")
            return


        self.raw_block=self.get_block()

        self.stage=1


        return self.raw_block

    def increment(self):
        self.counter+=1
        self.dict=self.on_edge()
        self.full = self.is_full()
        self.stage=0
        self.current_loc=self.get_raw_coordinates()

    def append(self):
        assert self.stage==1, "CALLED APPEND DURING FEED STAGE"
        self.blend_block_using_blend_function()
        #self.bump_blend()
        self.increment()

    def get_block(self):

        # initialize with zeros:
        raw_block = np.zeros(self.shape)

        cords = self.current_loc

        slice = self.raw[cords[0, 0]:cords[0, 1], cords[1, 0]:cords[1, 1], cords[2, 0]:cords[2, 1]]

        slice_shape = np.shape(slice)


        raw_block[0:slice_shape[0], 0:slice_shape[1], 0:slice_shape[2]] = slice

        return raw_block

    def get_raw_coordinates(self):

        dict_edges = self.on_edge()

        start_0 = 99999999999999999
        start_1 = 99999999999999999
        start_2 = 99999999999999999

        stop_0 = 99999999999999999
        stop_1 = 99999999999999999
        stop_2 = 99999999999999999

        # index notation
        dim_0 = self.counter % self.num_blocks[0]

        dim_1 = math.floor((self.counter % (self.num_blocks[0] * self.num_blocks[1])) / self.num_blocks[0])
        dim_2 = math.floor(self.counter / (self.num_blocks[0] * self.num_blocks[1]))

        # check positive edges and grab block from 0:size if contact
        if dict_edges["+0"]:
            start_0 = 0
            stop_0 = self.shape[0]

        if dict_edges["+1"]:
            start_1 = 0
            stop_1 = self.shape[1]

        if dict_edges["+2"]:
            start_2 = 0
            stop_2 = self.shape[2]

        # check negative edges and if true stop early to prevent going over edge
        if dict_edges["-0"]:
            start_0 = self.overlap_increment[0] * dim_0
            stop_0 = np.shape(self.raw)[0]

        if dict_edges["-1"]:
            start_1 = self.overlap_increment[1] * dim_1
            stop_1 = np.shape(self.raw)[1]

        if dict_edges["-2"]:
            start_2 = self.overlap_increment[2] * dim_2
            stop_2 = np.shape(self.raw)[2]

        # if neither are true assign index
        if dict_edges["+0"] == False and dict_edges["-0"] == False:
            start_0 = self.overlap_increment[0] * dim_0
            stop_0 = self.shape[0] + self.overlap_increment[0] * dim_0

        if dict_edges["+1"] == False and dict_edges["-1"] == False:
            start_1 = self.overlap_increment[1] * dim_1
            stop_1 = self.shape[1] + self.overlap_increment[1] * dim_1

        if dict_edges["+2"] == False and dict_edges["-2"] == False:
            start_2 = self.overlap_increment[2] * dim_2
            stop_2 = self.shape[2] + self.overlap_increment[2] * dim_2

        # grab slice and add to temp array
        start_0 = int(start_0)
        start_1 = int(start_1)
        start_2 = int(start_2)
        stop_0 = int(stop_0)
        stop_1 = int(stop_1)
        stop_2 = int(stop_2)

        coordinates = [[start_0, stop_0], [start_1, stop_1], [start_2, stop_2]]

        coordinates = np.asarray(coordinates, dtype=np.int)

        '''
        print("COUNTER:    %i" % self.counter)
        print("NUM BLOCKS: ", self.num_blocks)
        print(dict_edges)
        print("OVERLAP INCREMENT: ", self.overlap_increment)
        print("SLICE START:       ", [start_0, start_1, start_2])
        print("SLICE STOP:        ", [stop_0, stop_1, stop_2])
        print("DIMS:              ", [dim_0, dim_1, dim_2])
        print("AFF GRAPH SIZE:    ",np.shape(self.0))
        '''



        return coordinates

    def on_edge(self):
        '''
        Takes counter and num of blocks in each dim and determines if there is contact between the edge of the volume and the current block

        axis are traversed in order 0->1->2

        first go through all 0 values then increment 1 until all 1 values have been incremented and then 2

        if dict returned is true that means there is contact

        '''

        total_blocks = self.total_blocks

        dict = {
            "+0": False,
            "-0": False,
            "+1": False,
            "-1": False,
            "+2": False,
            "-2": False
        }


        if(self.num_blocks[0]!=1):
            # check +0
            if (self.counter % self.num_blocks[0] == 0):
                dict["+0"] = True

            # check -0
            if ((self.counter + 1) % self.num_blocks[0] == 0):
                dict["-0"] = True
        else:
            dict["+0"] = True
            dict["-0"] = True


        # check +1
        if (self.counter % (self.num_blocks[0] * self.num_blocks[1]) < self.num_blocks[0]):
            dict["+1"] = True

        # check -1
        if (self.counter % (self.num_blocks[0] * self.num_blocks[1])) + 1 > (self.num_blocks[0] * (self.num_blocks[1] - 1)):
            dict["-1"] = True

        # check +2
        if (self.counter < self.num_blocks[0] * self.num_blocks[1]):
            dict["+2"] = True

        # check -2
        if (self.counter + 1 > (self.num_blocks[0] * self.num_blocks[1] * (self.num_blocks[2] - 1))):
            dict["-2"] = True

        return dict

    def generate_aff(self):
        self.aff_block=self.function(self.raw_block)

    def process(self):

        while(self.full==False):
            assert type(self.function) != type([0,1]), "TOO MANY FUNCTIONS, CALL PROCESS AVRG INSTEAD"
            print("Block %i " %( self.counter+1), " of %i" % self.total_blocks)
            self.feed()
            self.aff_block=np.asarray(self.function(self.raw_block))
            self.append()
        self.crop_aff()
        return self.aff_graph

    def provide_all_raw_blocks(self):

        object_before=self

        self.overlap=0
        self.counter=0

        blocks=[]

        while (self.is_full() == False):
            self.feed()
            blocks.append(self.raw_block)
            self.increment()

        self.__dict__.update(object_before.__dict__)
        return blocks

    def process_avrg(self):
        assert type(self.function)==type([0,1]), "NO FUCTION LIST"

        len=np.shape(self.function)[0]
        while(self.full==False):
            print("Block %i " %( self.counter+1), " of %i" % self.total_blocks)
            self.feed()
            self.aff_block = np.multiply(np.asarray(self.function[0](self.raw_block)),1/len)
            for i in range(1,len):
                self.aff_block=np.add(self.aff_block,np.multiply(np.asarray(self.function[i](self.raw_block)),1/len))
            self.append()
        self.crop_aff()
        return self.aff_graph

    def calc_num_blocks(self):
        '''
            calculate raw dim  size using formula:

            size  =  (raw_shape)/(block_size)  *  (1/overlap)  -  1

            round up to get actual number

            calc padding with formula:

            padding  =  block_size/remainder


            padding is rounded up

            '''

        num_blocks_raw = np.zeros(3)
        num_blocks = np.zeros(3)
        padding = np.zeros(3)
        for x in range(0, 3):
            num_blocks_raw[x] = ((self.raw_shape[x]-self.shape[x])/self.overlap_increment[x])+1
            num_blocks[x] = int(math.ceil(num_blocks_raw[x]))
            remainder = round(num_blocks_raw[x] % 1, 4)
            padding[x] = int(math.ceil(remainder))

        return num_blocks, padding

    def is_full(self):
        if (self.counter < self.total_blocks):
            return False
        else:
            return True

    def blend_block_using_blend_function(self):


        #check if first in 0 axis
        if(self.dict["+0"]==True):
            self.initialize_buff0()


        #check if in middle of axis 0
        if(self.dict["+0"]==False and self.dict["-0"]==False):
            #if so blend on axis0
            self.blend_axis0_buf()

        #check if last in 0 axis but first in 1 axis
        if(self.dict["-0"]==True and self.dict["+1"]==True and self.dict["+0"]==False):
            #if so blend 0 and then initialize axis1 buffer with value
            self.blend_axis0_buf()
            self.initialize_buff1()

        #special case of 1 in this direction therfore no blending
        if (self.dict["-0"] == True and self.dict["+1"] == True and self.dict["+0"] == True):
            self.initialize_buff1()

        #check if last in 0 axis but middle of axis1
        if(self.dict["-0"]==True and self.dict["+1"]==False and self.dict["-1"]==False and self.dict["+0"]==False):
            #if so blend on to end of axis0 buffer then blend to axis1 buffer
            self.blend_axis0_buf()
            self.blend_axis1_buf()

        #check if last in 0 axis but middle of axis1 (special case of 1 in 0 axis)
        if(self.dict["-0"]==True and self.dict["+1"]==False and self.dict["-1"]==False and self.dict["+0"]==True):
            #if so blend on to end of axis0 buffer then blend to axis1 buffer
            self.blend_axis1_buf()

        #check if last in 0 and 1 but first in axis2
        if (self.dict["-0"] == True and self.dict["-1"] == True and self.dict["+2"] == True and self.dict["+0"]==False):
        #if so blend on axis 0 then blend on axis1 then initailize aff graph
            self.blend_axis0_buf()
            self.blend_axis1_buf()
            self.initialize_graph()

        # check if last in 0 and 1 but first in axis2 special case 0
        if (self.dict["-0"] == True and self.dict["-1"] == True and self.dict["+2"] == True and self.dict["+0"] == True):
            # if so blend on axis 0 then blend on axis1 then initailize aff graph
            self.blend_axis1_buf()
            self.initialize_graph()


        #check if last in 0 and 1 but middle of axis2
        if (self.dict["-0"] == True and self.dict["-1"] == True and self.dict["+2"] == False and self.dict["+0"]==False):
        #if so blend on axis 0 then blend on axis 1 then  blend on aff graph
            self.blend_axis0_buf()
            self.blend_axis1_buf()
            self.blend_axis2_graph()


        #special case
        if (self.dict["-0"] == True and self.dict["-1"] == True and self.dict["+2"] == False and self.dict["+0"]==True):
        #if so blend on axis 0 then blend on axis 1 then  blend on aff graph
            self.blend_axis1_buf()
            self.blend_axis2_graph()

    def initialize_buff0(self):
        self.buffer0 = np.zeros((3, np.shape(self.aff_graph)[1], self.shape[1], self.shape[2]))
        self.buffer0[:,0:self.shape[0],:,:]=self.aff_block

    def initialize_buff1(self):
        self.buffer1=np.zeros((3,np.shape(self.aff_graph)[1],np.shape(self.aff_graph)[2],self.shape[2]))
        self.buffer1[:,:,0:int(self.shape[1]),:]=self.buffer0
        self.buffer0 = np.zeros((3, np.shape(self.aff_graph)[1], self.shape[1], self.shape[2]))

    def initialize_graph(self):
        self.aff_graph[:,:,:,0:self.shape[2]]=self.buffer1

    def blend_axis0_buf(self):
        # blends current slice with preivous slice on axis0

        ##Go back one overlap increment to find start      ##take start and add the shape to it
        prev_slice = self.buffer0[:,
                     int(self.current_loc[0, 0] -self.overlap_increment[0]):int(self.current_loc[0, 0] - self.overlap_increment[0]+self.shape[0]),
                     0:self.shape[1],
                     0:self.shape[2]]


        blended_slice = blend.blend(prev_slice, self.aff_block, "0+", self.overlap, self.blend_fac)


        self.buffer0[:,
            int(self.current_loc[0, 0] - self.overlap_increment[0]):int(self.current_loc[0, 0] - self.overlap_increment[0] + np.shape(blended_slice)[1]),
            0:int(self.shape[1]),
            0:int(self.shape[2])] = blended_slice

    def blend_axis1_buf(self):

        #in output axis 0 is verticle , axis 1 is hor ect

        #blends content of axis0 buffer with prev slice of axis1 buffer and puts in axis1 buffer
        prev_slice=self.buffer1[:,:,int(self.current_loc[1,0]-self.overlap_increment[1]):int(self.current_loc[1,0]-self.overlap_increment[1]+self.shape[1]),:]

        blended_slice=blend.blend(prev_slice,self.buffer0,"1+",self.overlap,self.blend_fac)

        self.buffer1[:,:,int(self.current_loc[1,0]-self.overlap_increment[1]):int(self.current_loc[1,0]-self.overlap_increment[1]+np.shape(blended_slice)[2]),:]=blended_slice

    def blend_axis2_graph(self):

        prev_slice=self.aff_graph[:,:,:,int(self.current_loc[2,0]-self.overlap_increment[2]):int(self.current_loc[2,0]-self.overlap_increment[2]+self.shape[2])]

        blended_slice=blend.blend(prev_slice,self.buffer1,"2+",self.overlap,self.blend_fac)

        self.aff_graph[:,:,:,int(self.current_loc[2,0]-self.overlap_increment[2]):int(self.current_loc[2,0]-self.overlap_increment[2]+np.shape(blended_slice)[3])]=blended_slice

        #self.0[:,:,:,-20:]=1

    def crop_aff(self):
        self.aff_graph = np.asarray(self.aff_graph)[:, 0:self.raw_shape[0] , 0:self.raw_shape[1] , 0:self.raw_shape[2] ]












