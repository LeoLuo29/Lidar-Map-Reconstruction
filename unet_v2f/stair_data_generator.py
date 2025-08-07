## Given a matrix of coordination
## create a mask that blinds a desinated part of the data
## Feed the masked/blinded data to a DL model to learn the pattern
## the size and direction of the rectangular blinds are not uniform. 
## Generator_V1 generates stair data with univorm stair width and orientation
## Generator_V2 generates stair data with non univorm stair width; currently uniform orientation, but will be changed in the future


## start code 
import sys
import random
import numpy as np 
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
outfile = TemporaryFile()
from mpl_toolkits.mplot3d import Axes3D



class Blind():
    def __init__(self, x_length, y_length):
        self.mask_pnt_1 = []
        self.mask_pnt_2 = []
        self.mask_pnt_1.append(random.randint(5,x_length-5))
        self.mask_pnt_1.append(random.randint(5,y_length-5))
        self.mask_pnt_2.append(random.randint(self.mask_pnt_1[0],x_length-4))
        self.mask_pnt_2.append(random.randint(self.mask_pnt_2[0], y_length-4))


## create noise function
## currently not used...
class Noise():
    def __init__(self, noise_num, x_length, y_length):
        self.set = []
        for i in range (noise_num):
            temp_set = []
            temp_set.append(random.randint(1,x_length-1))
            temp_set.append(random.randint(1,y_length-1))
            self.set.append(temp_set) 





class Generator_V1():
    def __init__(self, set_num, x_length, y_length, stair_num, inpfilename, ansfilename):
        ##filename datatype is "" string; must end with .npy e.g.: "generated_input.npy"
        self.set_num, self.x_length, self.y_length, self.stair_num = set_num, x_length, y_length, stair_num
        self.totalnum = x_length * y_length * set_num
        self.inp = np.arange(self.totalnum).reshape(set_num,y_length,x_length)
        self.ans = np.arange(self.totalnum).reshape(set_num,y_length,x_length)

        ## building s sets of input datas
        for s in range(set_num):
            ##pnt_1 is the coordinate of top left masking corner; pnt_2 is the bottom right
            current_blind = Blind(x_length, y_length)
            self.mask_pnt_1 = current_blind.mask_pnt_1
            self.mask_pnt_2 = current_blind.mask_pnt_2 

            """
            self.mask_pnt_1 = []
            self.mask_pnt_2 = []
            self.mask_pnt_1.append(random.randint(12,x_length-12))
            self.mask_pnt_1.append(random.randint(12,y_length-12))
            self.mask_pnt_2.append(random.randint(self.mask_pnt_1[0],x_length-6))
            self.mask_pnt_2.append(random.randint(self.mask_pnt_2[0], y_length-6))
            """

            ## building simulated stairs 
            ## a -1 height meas that this part is masked/blinded
            self.height_increase = random.randint(4,8)
            self.stair_height = 0
            self.stair_width = y_length // stair_num
            for i in range(y_length):
                for j in range(x_length):
                    self.ans[s][i][j] = self.stair_height 
                    if (j >= self.mask_pnt_1[1] and j <= self.mask_pnt_2[1] and i >= self.mask_pnt_1[0] and i <= self.mask_pnt_2[0]):
                        self.inp[s][i][j] = -1
                    else:
                        self.inp[s][i][j] = self.stair_height
                if (i%self.stair_width == 0):
                        self.stair_height += self.height_increase


        np.save(inpfilename, self.inp)
        np.save(ansfilename, self.ans)
        ##breakpoint()
        ##print("ok it went through")
        print("data generated")
        return 


class ShowData():
     def __init__(self,inpfilename, ansfilename):
        fig = plt.figure(figsize=(64, 64))
        ax = fig.add_subplot(111, projection='3d')

        ##filename datatype is "" string
        self.inp = np.load(inpfilename)
        self.ans = np.load(ansfilename)

        ## printing data 
        self.full_array_string = np.array2string(self.inp, threshold = sys.maxsize)
        self.x_length = self.inp.shape[1]
        self.y_length = self.inp.shape[2]
        print(self.full_array_string)

        ## graphing data
        for i in range(self.y_length):
            for j in range(self.x_length):
                ax.scatter(i, j, self.inp[0][i][j], color='blue', marker='o', s=50, alpha=0.8)
        plt.show()




## Generator_V2 is a more complex stair data generator that output more versatile data interms of stiar's orientation,
## depth, width, and shape of blinds. 
class Generator_V2():
    def __init__(self, set_num, x_length, y_length, inpfilename, ansfilename):
        self.set_num, self.x_length, self.y_length = set_num, x_length, y_length
        self.totalnum = x_length * y_length * set_num
        self.inp = np.arange(self.totalnum).reshape(set_num,y_length,x_length)
        self.ans = np.arange(self.totalnum).reshape(set_num,y_length,x_length)

        for s in range(set_num):
            ## each set of stairs has a random orientation
            ## 0 faces foward, 1 faces backward
            ## 2 faces left up, 3 faces right up
            orientation = random.randint(0,4) 

            ## randomize the blind of stiars 
            current_blind = Blind(x_length, y_length)
            self.mask_pnt_1 = current_blind.mask_pnt_1
            self.mask_pnt_2 = current_blind.mask_pnt_2 

            ## storing the width of each stair into a list
            self.stair_widths = self.create_stair_widths(x_length = self.x_length)

            ##
            self.stair_height = 0 
            i = 0 
            for w in self.stair_widths:
                for widths_loop_iterater in range(w):
                    for j in range(x_length):
                        ## blinded 
                        if self.blind_check(i,j):
                            self.inp[s][i][j] = -1
                        
                        ## non blinded
                        else:
                            self.inp[s][i][j] = self.stair_height
                        ## update the answer after inp updated
                        self.ans[s][i][j] = self.stair_height
                    i +=1 
                self.stair_height += 5 
        
        ##
        np.save(inpfilename, self.inp)
        np.save(ansfilename, self.ans)
        print("data generated; v2")
    

    
    ## if return true, blinb current point with -1
    def blind_check(self, i, j):
        return j >= self.mask_pnt_1[1] and j <= self.mask_pnt_2[1] and i >= self.mask_pnt_1[0] and i <= self.mask_pnt_2[0]
    
    ## generate in advace a list of width that has various stair width from 2~5.
    ## however, the total width adding together perfectly matches the exact total length by adjusting the lenth of the las stair
    def create_stair_widths(self, x_length):
        stair_widths = []
        remaining = x_length 
        while True:
            temp_width = random.randint(2,5)
            if ((remaining - temp_width) < 0):
                stair_widths.append(remaining)
                break
            
            ##
            stair_widths.append(temp_width)
            remaining -= temp_width
        
        return stair_widths




## produces randomlized stair depth & width, multiple stiar heights, and granular like noise
class Generator_V3():
    def __init__(self, set_num, x_length, y_length, noise_num, inpfilename, ansfilename):
        self.set_num, self.x_length, self.y_length, self.noise_num = set_num, x_length, y_length, noise_num
        self.totalnum = x_length * y_length * set_num
        self.inp = np.arange(self.totalnum).reshape(set_num,y_length,x_length)
        self.ans = np.arange(self.totalnum).reshape(set_num,y_length,x_length)

        for s in range(set_num):

            ## randomize the blind of stiars 
            current_blind = Blind(x_length, y_length)
            self.mask_pnt_1 = current_blind.mask_pnt_1
            self.mask_pnt_2 = current_blind.mask_pnt_2 

            ## storing the width of each stair into a list
            self.stair_widths = self.create_stair_widths(x_length = self.x_length)

            ##
            self.stair_height = 0 
            i = 0 
            for w in self.stair_widths:
                for widths_loop_iterater in range(w):
                    for j in range(x_length):
                        ## blinded 
                        if self.blind_check(i,j):
                            self.inp[s][i][j] = -1
                        
                        ## non blinded
                        else:
                            self.inp[s][i][j] = self.stair_height
                        ## update the answer after inp updated
                        self.ans[s][i][j] = self.stair_height
                    i +=1 
                self.stair_height += 0.2

            ## create noise
            self.create_noise(s)
        
        ##
        np.save(inpfilename, self.inp)
        np.save(ansfilename, self.ans)
        print("data generated; v3")


    ## function called in each set to create noise
    def create_noise(self,current_set):
        for i in range(self.noise_num):
            cur_x = random.randint(4,self.x_length-3)
            cur_y = random.randint(4,self.y_length-3)
            if self.blind_check(cur_y,cur_x): continue
            noise_height = random.randint(self.inp[current_set][cur_y][cur_x],10) 
            self.inp[current_set][cur_y][cur_x] = noise_height



    ## if return true, blinb current point with -1
    def blind_check(self, i, j):
        return j >= self.mask_pnt_1[1] and j <= self.mask_pnt_2[1] and i >= self.mask_pnt_1[0] and i <= self.mask_pnt_2[0]
    
    

    ## generate in advace a list of width that has various stair width from 2~5.
    ## however, the total width adding together perfectly matches the exact total length by adjusting the lenth of the las stair
    def create_stair_widths(self, x_length):
        stair_widths = []
        remaining = x_length 
        while True:
            temp_width = random.randint(2,5)
            if ((remaining - temp_width) < 0):
                stair_widths.append(remaining)
                break
            
            ##
            stair_widths.append(temp_width)
            remaining -= temp_width
        
        return stair_widths




## Generator() & ShowData() argument format: "folder_name/file_name"
## G = Generator_V1(100,20,20,3,"train_data/generated_inp_easy_train.npy","train_data/generated_ans_easy_train.npy")
## G = Generator_V2(100,64,64,"test_data2/generated_inp_test1.npy","test_data2/generated_ans_test1.npy")
G = Generator_V3(100,64,64,100,"test_data2/generated_inp_test2.npy","test_data2/generated_ans_test2.npy")
S = ShowData("test_data2/generated_inp_test2.npy", "test_data2/generated_ans_test2.npy")  


