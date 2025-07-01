import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.measurements import center_of_mass

class Tensor():
    
    def __init__(self, Exx, Exy, Exz, Eyx, Eyy, Eyz, Ezx, Ezy, Ezz):
        
        self.E1, self.E2, self.E3 = Exx.copy(), Exy.copy(), Exz.copy()
        self.E4, self.E5, self.E6 = Eyx.copy(), Eyy.copy(), Eyz.copy()
        self.E7, self.E8, self.E9 = Ezx.copy(), Ezy.copy(), Ezz.copy()

    def asmat(self):
        return np.array([[self.E1,self.E2,self.E3],
                         [self.E4,self.E5,self.E6],
                         [self.E7,self.E8,self.E9]]).transpose((2,3,4,0,1))
        
    def asvoigt(self):
        return self.E1, self.E2, self.E3, self.E4, self.E5, self.E6, self.E7, self.E8, self.E9 
        
    def transpose(self):
        return Tensor(self.E1, self.E4, self.E7, self.E2, self.E5, self.E8, self.E3, self.E6, self.E9)
    
    def identity_add(self):
        self.E1 += 1; self.E5 += 1; self.E9 += 1 
    
    def identity_subtract(self):
        self.E1 -= 1; self.E5 -= 1; self.E9 -= 1 
        
    @staticmethod
    def dot(X, Y):
        
        X1,X2,X3,X4,X5,X6,X7,X8,X9=X.asvoigt()
        Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9=Y.asvoigt()
        
        Z1, Z2, Z3 = X1*Y1 + X2*Y4 + X3*Y7, X1*Y2 + X2*Y5 + X3*Y8, X1*Y3 + X2*Y6 + X3*Y9
        Z4, Z5, Z6 = X4*Y1 + X5*Y4 + X6*Y7, X4*Y2 + X5*Y5 + X6*Y8, X4*Y3 + X5*Y6 + X6*Y9
        Z7, Z8, Z9 = X7*Y1 + X8*Y4 + X9*Y7, X7*Y2 + X8*Y5 + X9*Y8, X7*Y3 + X8*Y6 + X9*Y9
        
        return Tensor(Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9)
    

class MyocardialStrain():
    
    def __init__(self, mask, flow):
                
        self.mask  = mask
        self.flow  = flow
        
        assert len(mask.shape) == 3
        assert len(flow.shape) == 4
        assert mask.shape == flow.shape[:3]
        assert flow.shape[-1] == 3

    def calculate_strain(self, lv_label=3):
        
        cx, cy, cz = center_of_mass(self.mask==lv_label)
        nx, ny, nz = self.mask.shape
        
        self.flow_rot = roll_to_center(self.flow, cx, cy)
        self.mask_rot = roll_to_center(self.mask, cx, cy)

        ux, uy, uz  = np.array_split(self.flow_rot, 3, -1)
        Uxx,Uxy,Uxz = np.gradient(np.squeeze(ux))
        Uyx,Uyy,Uyz = np.gradient(np.squeeze(uy))
        Uzx,Uzy,Uzz = np.gradient(np.squeeze(uz))
        
        F = Tensor(Uxx,Uxy,Uxz,Uyx,Uyy,Uyz,Uzx,Uzy,Uzz)

        F.identity_add()
        F = F.dot(F.transpose(), F) 
        F.identity_subtract()
        
        self.Err, self.Ecc, self.Erc, self.Ecr = convert_to_polar(mask=self.mask_rot, E=0.5*F.asmat()[:,:,:,:2,:2])
        

def roll(x, rx, ry):
    x = np.roll(x, rx, axis=0)
    return np.roll(x, ry, axis=1)

def roll_to_center(x, cx, cy):
    nx, ny = x.shape[:2]
    return roll(x,  int(nx//2-cx), int(ny//2-cy))

def polar_grid(nx=128, ny=128):
    x, y = np.meshgrid(np.linspace(-nx//2, nx//2, nx), np.linspace(-ny//2, ny//2, ny))
    phi  = (np.rad2deg(np.arctan2(y, x)) + 180).T
    r    = np.sqrt(x**2+y**2+1e-8)
    return phi, r

def convert_to_polar(mask, E):

    phi = polar_grid(*E.shape[:2])[0]
    Err = np.zeros(mask.shape)
    Ecc = np.zeros(mask.shape)
    Erc = np.zeros(mask.shape)
    Ecr = np.zeros(mask.shape)
    for k in range(mask.shape[-1]):
        cos = np.cos(np.deg2rad(phi))
        sin = np.sin(np.deg2rad(phi))

        Exx, Exy, Eyx, Eyy = E[:,:,k,0,0],E[:,:,k,0,1],E[:,:,k,1,0],E[:,:,k,1,1] # convert cartesian (Exx, Exy, Eyx, Eyy) to polar
        Err[:,:,k] +=  cos*( cos*Exx+sin*Exy) + sin*( cos*Eyx+sin*Eyy)
        Ecc[:,:,k] += -sin*(-sin*Exx+cos*Exy) + cos*(-sin*Eyx+cos*Eyy)
        Erc[:,:,k] +=  cos*(-sin*Exx+cos*Exy) + sin*(-sin*Eyx+cos*Eyy)
        Ecr[:,:,k] += -sin*( cos*Exx+sin*Exy) + cos*( cos*Eyx+sin*Eyy)

    return Err, Ecc, Erc, Ecr

    
# def convert_to_aha4d(tensor, mask):
#     Tensor = tensor.copy()
#     Mask   = mask.copy()

#     Tensor[Mask!=2,:] = 0

#     # rotate to have RV center of mass on the right
#     angle  = _get_lv2rv_angle(Mask)
#     Tensor = rotate(Tensor,  -angle, reshape=False, order=0)
#     Mask   = rotate(Mask, -angle, reshape=False, order=1).clip(0,3).round()

#     # roll to center 
#     cx, cy = center_of_mass(Mask>1)[:2]
#     Tensor = np.flipud(np.rot90(_roll_to_center(Tensor, cx, cy)))
#     Mask   = np.flipud(np.rot90(_roll_to_center(Mask, cx, cy)))

#     # remove slices that do not contain tissue labels
#     ID     = Mask.sum(axis=(0,1))>0
#     Mask   = Mask[:,:,ID]
#     Tensor = Tensor[:,:,ID]
    
#     return Tensor, Mask

class Rotate_data():
    def __init__(self, Err, Ecc, mask,img, insertion_p1, insertion_p2):
        self.Err  = Err
        self.Ecc  = Ecc
        self.mask = mask
        self.img  = img
        self.insertion_p1 = insertion_p1
        self.insertion_p2 = insertion_p2
        # self.non_slice_num = non_slice_num

    def rotate_orientation(self , for_visualization = False):
        # goal: RV is at left of LV
        # if for_visualization: the center of septum = 90 degree so that the center of anterior = 0 degree
        # if not for visualization (for calculation and aha plot): the center of septum = 120 degree so that the boundary of anterior = 0 degree (thus the first segment 0-60 is anterior)
        Err  = self.Err.copy()
        Ecc  = self.Ecc.copy()
        mask = self.mask.copy()
        img = self.img.copy()

        Err[mask!=2] = 0
        Ecc[mask!=2] = 0

        angle,_,_,_,_ = _get_lv2rv_angle_using_insertion_points(mask, self.insertion_p1, self.insertion_p2)
        if for_visualization:  # RVthe center of RV is 
            Ecc   = rotate(Ecc,  -angle, reshape=False, order=1)
            Err   = rotate(Err,  -angle, reshape=False, order=1)
            mask  = rotate(mask, -angle, reshape=False, order=0).clip(0,3).round()  
            img = rotate(img, -angle, reshape=False, order=2)
        else: # for calculation and AHA plot
            Ecc   = rotate(Ecc,  -angle + 30, reshape=False, order=1)
            Err   = rotate(Err,  -angle + 30, reshape=False, order=1)
            mask  = rotate(mask, -angle + 30, reshape=False, order=0).clip(0,3).round()
            img = rotate(img, -angle + 30, reshape=False, order=2)
        
        
        # roll to center o
        cx, cy = center_of_mass(mask>1)[:2]
        Ecc_rot  = np.rot90(np.rot90(_roll_to_center(Ecc, cx, cy)))
        Err_rot  = np.rot90(np.rot90(_roll_to_center(Err, cx, cy)))
        mask_rot = np.rot90(np.rot90(_roll_to_center(mask, cx, cy))) 
        img_rot = np.rot90(np.rot90(_roll_to_center(img, cx, cy)))

        # # remove slices that do not contain tissue labels
        # ID   = self.non_slice_num
        # mask_rot = mask_rot[:,:,ID]
        # Err_rot  = Err_rot[:,:,ID]
        # Ecc_rot  = Ecc_rot[:,:,ID]
        return Err_rot, Ecc_rot, mask_rot, img_rot#, ID

    
class PolarMap():
    
    def __init__(self, Err, Ecc, mask):       
        self.Err  = Err
        self.Ecc  = Ecc
        self.mask = mask

    def project_to_aha_polar_map(self):

        Err = self.Err.transpose((2,0,1))
        Ecc = self.Ecc.transpose((2,0,1))
        print('... radial strain')
        V_err = self._project_to_aha_polar_map(Err)
        print('... circumferential strain')
        V_ecc = self._project_to_aha_polar_map(Ecc)

        results = {'V_err':V_err, 'V_ecc':V_ecc, 'mask':self.mask}

        return results
        
    def _project_to_aha_polar_map(self, E, nphi=360, nrad=100, dphi=1):
        
        nz = E.shape[0]
        angles = np.arange(0, nphi, dphi)
        V      = np.zeros((nz, angles.shape[0], nrad))

        for rj in range(nz):

            m = np.copy(self.mask[:,:,rj])
            cx, cy = center_of_mass(m==2)

            E_rj = E[rj]
            E_rj = roll_to_center(E_rj, cx, cy) # center the image
            Err_q  = _inpter2(E_rj) # increase size by 10

            # Err_q  = _inpter2(E[rj]) # increase size by 10

            PHI, R = _polar_grid(*Err_q.shape) # *Err_q.shape as well as PHI and R = (1280, 1280)
            PHI = PHI.ravel() # flatten
            R   = R.ravel()

            for k, pmin in enumerate(angles):
                pmax = pmin + 3  # large increment to include more points in myocardium # previous dphi/2.0 # 0.5 increment

                # Get values for angle segment
                PHI_SEGMENT = (PHI>=pmin)&(PHI<=pmax)
                Rk   = R[PHI_SEGMENT]
                PHIk = PHI[PHI_SEGMENT]
                Vk   = Err_q.ravel()[PHI_SEGMENT]
                
                # only use the one with non-zero values in Rk and Vk
                Rk = Rk[np.abs(Vk)!=0]
                Vk = Vk[np.abs(Vk)!=0]
                
                if len(Vk) == 0 or len(Vk) == 1:
                    continue # this might not be the best
                Rk = _rescale_linear(Rk, rj, rj + 1)  #scale Rk into 0 to 1

                r = np.arange(rj, rj+1, 1.0/nrad)  # divide Rk into nrad parts
                f = interp1d(Rk, Vk) # need to interpolate Vk using the known Vk in Rk
                v = f(r)

                V[rj,k] += v

        return V        
        
        
    def construct_AHA_map(self, tensor, start_slice_name, start=20, stop=80, sigma=12):

        E  = tensor.copy()
        mu = E[:,:,start:stop].mean()

        nz = E.shape[0]
        E  = np.concatenate(np.array_split(E[:,:,start:stop], nz), axis=-1)[0] # stack the slices together,  shape (360, (stop - start) * slice_num)

        old = E.shape[1]/nz*1. # original R (stop - start)
        for j in range(nz-1):
            xi = int(old//2+j*old)
            xj = int(old+old//2+j*old)
            E[:,xi:xj] = gaussian_filter(E[:,xi:xj],sigma=sigma, mode='wrap')
            E[:,xi:xj] = gaussian_filter(E[:,xi:xj],sigma=sigma, mode='wrap')

        E = np.stack(np.array_split(E,nz,axis=1)) # put into original shape

        # divide into apical, mid and basal layers
        slices_per_layer = E.shape[0]//3
        mod = E.shape[0] % 3
        if mod == 1 or mod == 2:
            mod = 1

        if start_slice_name == 'apex':
            layer1 = E[0:slices_per_layer,...]
            layer2 = E[slices_per_layer : slices_per_layer * 2 + mod,...]
            layer3 = E[slices_per_layer * 2 + mod: E.shape[0],...]
        
        else: # start from "base"
            layer1 = E[E.shape[0] - slices_per_layer :E.shape[0],...]
            layer2 = E[E.shape[0] - slices_per_layer * 2 - mod : E.shape[0] - slices_per_layer,...]
            layer3 = E[0: E.shape[0] - slices_per_layer * 2 -mod,...]

        E = [layer1, layer2, layer3]  ######### apical, mid, basal

        E = [np.mean(E[i], axis=0) for i in range(3)] # calculate mean across all slices in each layer
        E = np.concatenate(E, axis=1)  # shape (360, 3 * (stop - start))

        mu = [mu] + self._get_16segments(E) +[0]

        return E, mu 
    
    def _get_16segments(self, data):
        c2,c3,c4 = np.array_split(data,3,axis=-1)

        c4 = [np.mean(ci) for ci in np.array_split(c4,6,axis=0)]  # basal
     
        c3 = [np.mean(ci) for ci in np.array_split(c3,6,axis=0)]  # mid

        c2 = [np.mean(ci) for ci in np.array_split(c2,4,axis=0)]  # apical

        c = c4 + c3 + c2 
        return c

    
    def _get_17segments(self, data):
        c1,c2,c3,c4 = np.array_split(data,4,axis=-1)
        c2 = np.roll(c2,-45,0)
        #c2 = np.roll(c2,-90,0)

        c4 = [np.mean(ci) for ci in np.array_split(c4,6,axis=0)]  # basal
        c4 = list(np.roll(np.array(c4),-1))
        c3 = [np.mean(ci) for ci in np.array_split(c3,6,axis=0)]  # mid
        c3 = list(np.roll(np.array(c3),-1))
        c2 = [np.mean(ci) for ci in np.array_split(c2,4,axis=0)]  # apical
        #c2 = list(np.roll(np.array(c2),-1))
        c1 = [np.mean(c1)]  # apex

        c = c4 + c3 + c2 + c1  # basal, mid, apical, apex

        return c

    def _get_17segments_RC(self, data1,data2):
        
        def _rc(a,b):
            #return np.mean(np.abs((b-a)/b)*100)
            return np.mean(((b-a)/b)*100)
        
        c1_1,c2_1,c3_1,c4_1 = np.array_split(data1,4,axis=-1)
        c1_2,c2_2,c3_2,c4_2 = np.array_split(data2,4,axis=-1)

        c4 = [_rc(ci1,ci2) for ci1,ci2 in zip(np.array_split(c4_1,6,axis=0),
                                              np.array_split(c4_2,6,axis=0))]
        c4 = list(np.roll(np.array(c4),-1))
        
        c3 = [_rc(ci1,ci2) for ci1,ci2 in zip(np.array_split(c3_1,6,axis=0),
                                              np.array_split(c3_2,6,axis=0))]
        c3 = list(np.roll(np.array(c3),-1))
        
        c2 = [_rc(ci1,ci2) for ci1,ci2 in zip(np.array_split(c2_1,4,axis=0),
                                              np.array_split(c2_2,4,axis=0))]
        c2 = list(np.roll(np.array(c2),-1))
        c1 = [_rc(c1_1,c1_2)]

        c = c4 + c3 + c2 + c1

        return c
    

class wall_thickness_change_index():
   def __init__(self, mask_rot, mask_rot_es):       
      self.mask_tf1  = mask_rot
      self.mask_tf2  = mask_rot_es

   def calculate_index(self):
      # prepare data
      E = np.copy(self.mask_tf1)
      E_es = np.copy(self.mask_tf2)

      # get len change
      len_change_tf1 = self.calculate_len_change(E)
      len_change_tf2 = self.calculate_len_change(E_es)

      wtci = np.zeros(len_change_tf1.shape)
      for i in range(len_change_tf1.shape[0]):
         for j in range(len_change_tf1.shape[1]):
           if len_change_tf1[i,j,0] != 0:
            wtci[i,j] += (len_change_tf2[i,j,0] - len_change_tf1[i,j,0])/len_change_tf1[i,j,0]
         
      return wtci

      
   def calculate_len_change(self, E):
      '''E should be equal to np.copy(self.mask_tf)'''
      # prepare data
      E = E.transpose((2,0,1))
 
      # prepare angles and radians
      nz = E.shape[0]
      angles = np.arange(0, 360, 1)
      len_change = np.zeros((nz, angles.shape[0], 100))

      for rj in range(nz):
     
         # find the center of mass
         m = np.copy(self.mask_tf1[:,:,rj])
         cx, cy = center_of_mass(m>=2)  

         # find the slice
         E_rj = E[rj]
         E_rj = roll_to_center(E_rj, cx, cy)

         Err_q  = _inpter2(E_rj)
         Err_q[abs(Err_q - 2) < 0.3] = 2 # stablize

         PHI, _ = _polar_grid(*Err_q.shape) # *Err_q.shape as well as PHI and R = (1280, 1280)
         PHI = PHI.ravel() # flatten

         total_dis = 0
         for k, pmin in enumerate(angles):

            pmax = pmin + 3 # 3 for stablization
            PHI_SEGMENT = (PHI>=pmin)&(PHI<=pmax)
            PHI_SEGMENT = np.reshape(PHI_SEGMENT, (1280, 1280))

            points = np.where(PHI_SEGMENT==True)

            # let's find out several boundary points
            # first, in "points", which point is the cloest to the [cx, cy]
            dis = np.sqrt((points[0]-cx * 10)**2 + (points[1]-cy * 10)**2)
            dis_index = np.argsort(dis)
            innest_point = points[0][dis_index[0]], points[1][dis_index[0]]

            # second, let's find out which points in "points" have value equal to 2 in Err_q
            points_value = Err_q[points[0], points[1]]
            points_value_index = np.where(points_value==2)
            points_value_index = points_value_index[0]
            points_w_2 = points[0][points_value_index], points[1][points_value_index]

            # third, find out in points_w_2, which point is the cloest to the innest_point
            point_dis = np.sqrt((points_w_2[0]-innest_point[0])**2 + (points_w_2[1]-innest_point[1])**2)
            if point_dis.shape[0] != 0:
                dis_index = np.argsort(point_dis)
                start_point = points_w_2[0][dis_index[0]], points_w_2[1][dis_index[0]]

                # fourth, find out in points_w_2, which point is further to the innest_point
                # dis = np.sqrt((points_w_2[0]-innest_point[0])**2 + (points_w_2[1]-innest_point[1])**2)
                # dis_index = np.argsort(dis)
                end_point = points_w_2[0][dis_index[-1]], points_w_2[1][dis_index[-1]]

                # fifth, calculat the euclidean distance between start_point and end_point
                total_dis = np.sqrt((start_point[0]-end_point[0])**2 + (start_point[1]-end_point[1])**2)
             
            else:
                total_dis = total_dis # use the previous angle


            # plot if needed
            # plt.figure(figsize=(8,4))
            # plt.subplot(121); plt.imshow(Err_q, cmap='gray')
            # PHI = np.reshape(PHI, (1280, 1280))
            # Err_q[innest_point[0] :innest_point[0] + 5 , innest_point[1] : innest_point[1] +5] = 5
            # Err_q[start_point[0] :start_point[0] + 5 , start_point[1] : start_point[1] +5] = 5
            # Err_q[end_point[0] :end_point[0] + 5 , end_point[1] : end_point[1] +5] = 5
            # plt.subplot(122); plt.imshow(Err_q , cmap='gray')

            len_change[rj,k] += total_dis

      return len_change 
   
   def construct_AHA_map(self, tensor, start_slice_name, sigma = 12):

        E  = tensor.copy()
        mu = E[:,:,:].mean()

        nz = E.shape[0]
        E  = np.concatenate(np.array_split(E[:,:,:], nz), axis=-1)[0] # stack the slices together,  shape (360, (stop - start) * slice_num)

        old = E.shape[1]/nz*1. # original R (stop - start)
        for j in range(nz-1):
            xi = int(old//2+j*old)
            xj = int(old+old//2+j*old)
            E[:,xi:xj] = gaussian_filter(E[:,xi:xj],sigma=sigma, mode='wrap')
            E[:,xi:xj] = gaussian_filter(E[:,xi:xj],sigma=sigma, mode='wrap')

        E = np.stack(np.array_split(E,nz,axis=1)) # put into original shape

        # divide into apical, mid and basal layers
        slices_per_layer = E.shape[0]//3
        mod = E.shape[0] % 3
        if mod == 1 or mod == 2:
            mod = 1

        if start_slice_name == 'apex':
            layer1 = E[0:slices_per_layer,...]
            layer2 = E[slices_per_layer : slices_per_layer * 2 + mod,...]
            layer3 = E[slices_per_layer * 2 + mod: E.shape[0],...]
        
        else: # start from "base"
            layer1 = E[E.shape[0] - slices_per_layer :E.shape[0],...]
            layer2 = E[E.shape[0] - slices_per_layer * 2 -mod : E.shape[0] - slices_per_layer,...]
            layer3 = E[0: E.shape[0] - slices_per_layer * 2 -mod,...]

        E = [layer1, layer2, layer3]  ######### apical, mid, basal
        # print('layer1, layer2, layer3: ', layer1.shape, layer2.shape, layer3.shape)

        E = [np.mean(E[i], axis=0) for i in range(3)] # calculate mean across all slices in each layer
        # print('E: ', E[0].shape, E[1].shape, E[2].shape)
        E = np.concatenate(E, axis=1)  # shape (360, 3 * (stop - start))

        mu = [mu] + self._get_16segments(E) +[0]

        return E, mu 
    
   def _get_16segments(self, data):
        c2,c3,c4 = np.array_split(data,3,axis=-1)

        c4 = [np.mean(ci) for ci in np.array_split(c4,6,axis=0)]  # basal
     
        c3 = [np.mean(ci) for ci in np.array_split(c3,6,axis=0)]  # mid

        c2 = [np.mean(ci) for ci in np.array_split(c2,4,axis=0)]  # apical

        c = c4 + c3 + c2 
        return c
   

class Find_AHA_segment_centers():
    def __init__(self, mask_rot):       
        self.mask = mask_rot

    def assign_different_values_for_each_aha(self):
        E = np.copy(self.mask)
        nz = E.shape[-1]
        angles = np.arange(0, 360, 1)

        angle_intervals = [[-30,30], [30,90], [90,150], [150,210], [210,270], [270,330]]
        angle_intervals_apex = [[-45,45], [45,135],[135,225], [225,315]]

        E_aha = np.zeros_like(E)

        for rj in range(0, nz):
            E_slice = np.copy(E[:,:, rj])

            # find the center of mass and move to center
            cx, cy = center_of_mass(np.copy(E_slice)>=2)  
            if np.isnan(cx) == 1:
                continue
            
            nx, ny = np.copy(E_slice).shape[:2]
            E_slice_centered = roll(np.copy(E_slice),  int(nx//2-cx), int(ny//2-cy))

            E_slice_centered_aha = np.copy(E_slice_centered)

            PHI, _ = _polar_grid(*E_slice_centered.shape) # *Err_q.shape as well as PHI and R = (128,128)
            PHI = PHI.ravel() # flatten

            if rj < 6: # base and mid
                angle_intervals_now = angle_intervals
            else: # apex
                angle_intervals_now = angle_intervals_apex

            # assign aha pixels
            for k, angle_interval in enumerate(angle_intervals_now):
        
                p_min = angle_interval[0]
                p_max = angle_interval[1]

                if p_min == -30:
                    PHI_SEGMENT = np.logical_or(np.logical_and(PHI >= 0, PHI <= 30), np.logical_and(PHI >= 330, PHI <= 360))
            
                elif p_min == -45:
                    PHI_SEGMENT = np.logical_or(np.logical_and(PHI >= 0, PHI <= 45), np.logical_and(PHI >= 315, PHI <= 360))
                else:
                    PHI_SEGMENT = (PHI>=p_min)&(PHI<=p_max)

                PHI_SEGMENT = np.reshape(PHI_SEGMENT, (128, 128))

                points = np.where(PHI_SEGMENT==True)

                for p in range(len(points[0])):
                    if E_slice_centered_aha[points[0][p], points[1][p]] == 2:
                        E_slice_centered_aha[points[0][p], points[1][p]] = k + 5  ####### different values!!!

            # move back
            E_slice_aha = roll(np.copy(E_slice_centered_aha),  -int(nx//2-cx), -int(ny//2-cy))
            E_aha[:,:,rj] = E_slice_aha  

        return E_aha
    
    def find_aha_segment_centers(self, E_aha):
        # find the center of each AHA segmentation
        region_list = ['base', 'mid', 'apex']
        aha_center_list_int = np.zeros([16,3])
        aha_center_list_decimals = np.zeros([16,3])

        # let's do base first
        for region in region_list:
            if region[0:2] == 'ba': # base
                slice_start = 0
                slice_end = 3
                aha_index = [1,2,3,4,5,6]
            elif region[0:2] == 'mi': # mid
                slice_start = 3
                slice_end = 6
                aha_index = [7,8,9,10,11,12]
            elif region[0:2] == 'ap': # apex
                slice_start = 6
                slice_end = 9
                aha_index = [13,14,15,16]
            
            for index in range(len(aha_index)):
                E_aha_region = np.copy(E_aha[:,:, slice_start : slice_end])

                E_aha_specific_segment = np.copy(E_aha_region) == (5 + index)
                center_specific_segment = center_of_mass(E_aha_specific_segment)

                aha_center_list_int[aha_index[index]-1, 0] = int(np.round(center_specific_segment[0]))
                aha_center_list_int[aha_index[index]-1, 1] = int(np.round(center_specific_segment[1]))
                aha_center_list_int[aha_index[index]-1, 2] = int(slice_start + np.round(center_specific_segment[2]))

                aha_center_list_decimals[aha_index[index]-1, 0] = center_specific_segment[0]
                aha_center_list_decimals[aha_index[index]-1, 1] = center_specific_segment[1]
                aha_center_list_decimals[aha_index[index]-1, 2] = slice_start + center_specific_segment[2]



        E_aha_w_centers = np.copy(E_aha)
        for aha_center in aha_center_list_int:
            E_aha_w_centers[int(aha_center[0]), int(aha_center[1]), int(aha_center[2])] = 12

        return aha_center_list_int, aha_center_list_decimals, E_aha_w_centers


    
def _roll(x, rx, ry):
    x = np.roll(x, rx, axis=0)
    return np.roll(x, ry, axis=1)

def _roll_to_center(x, cx, cy):
    nx, ny = x.shape[:2]
    return _roll(x,  int(nx//2-cx), int(ny//2-cy))

def _py_ang(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'. """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.rad2deg(np.arctan2(sinang, cosang))

def _polar_grid(nx=128, ny=128):
    x, y = np.meshgrid(np.linspace(-nx//2, nx//2, nx), np.linspace(-ny//2, ny//2, ny))
    phi  = (np.rad2deg(np.arctan2(y, x)) + 180).T
    r    = np.sqrt(x**2+y**2+1e-8)
    return phi, r

def _rescale_linear(array, new_min, new_max):
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max-new_min) / (maximum-minimum)
    b = new_min - m* minimum
    return m*array + b

def _inpter2(Eij, k=10):
    nx, ny = Eij.shape
    
    x  = np.linspace(0,nx-1,nx)
    y  = np.linspace(0,ny-1,ny)
    xq = np.linspace(0,nx-1,nx*k)
    yq = np.linspace(0,ny-1,ny*k)
    
    f = interp2d(x,y,Eij,kind='linear')
    
    return f(xq,yq)    
    
def _get_lv2rv_angle_using_mask(mask):
    cx_lv, cy_lv = center_of_mass(mask>1)[:2]
    cx_rv, cy_rv = center_of_mass(mask==1)[:2]
    phi_angle    = _py_ang([cx_rv-cx_lv, cy_rv-cy_lv], [0, 1])
    return phi_angle  

def _get_lv2rv_angle_using_insertion_points(mask, insertion_p1, insertion_p2):
    cx_lv, cy_lv = center_of_mass(mask>1)[:2]
    cx_rv, cy_rv = (insertion_p1[0] + insertion_p2[0])//2 , (insertion_p1[1] + insertion_p2[1])//2
    phi_angle    = _py_ang([cx_rv-cx_lv, cy_rv-cy_lv], [0, 1])
    return phi_angle   , cx_lv, cy_lv, cx_rv, cy_rv 
    
    
    
### FUNCTIONS TO PLOT THE POLAR MAP 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

def _write(ax, mu, j, theta_i, i, width=2):
    xi, yi = polar2cart(0,  theta_i)
    xf, yf = polar2cart(35, theta_i)

    l = Line2D([40-xi,40-xf], [40-yi,40-yf], color='black', linewidth=width)
    ax.add_line(l)
    xi, yi = polar2cart(30, theta_i + 2*np.pi/12)
    ax.text(40-xi-.3, 40-yi, '%d' %(mu[j][i]), weight='bold', fontsize=14)
    
def write(ax, mu, j, width=2):
    if j > 1:
        for i in range(6):
            theta_i = 2*np.pi - i*60*np.pi/180 + 2*60*np.pi/180
            _write(ax, mu, j, theta_i, i)
    if j == 1:
        for i in range(4):
            theta_i = i*90*np.pi/180 - 45*np.pi/180
            _write(ax, mu, j, theta_i, i)
    if j ==0:
        ax.text(40-.3, 40, '%d' %(mu[j][0]), weight='bold', fontsize=14)

def plot_bullseye(data,mu,vmin=None,vmax=None, savepath=None,cmap='RdBu_r', label='GPRS (%)', 
                  std=None,cbar=False,color='white', fs=20, xshift=0, yshift=0, ptype='mesh',frac=False):
    
    rho     = np.arange(0,4,4.0/data.shape[1])
    Theta   = np.deg2rad(range(90, data.shape[0] + 90))
    [th, r] = np.meshgrid(Theta, rho)

    fig, ax = plt.subplots(figsize=(6,6))
    #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    #ax.axis('tight') creates errors 
    #ax.axis('off')
    if ptype == 'mesh':
        im = ax.pcolormesh(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    else:
        im = ax.contourf(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    if cbar:
        cbar = plt.colorbar(im, cax=fig.add_axes([0.15, -0.03, 0.7, 0.05]), orientation='horizontal')

        new_ticks = []
        new_ticks_labels = []
        for i,tick in enumerate(cbar.ax.get_xticks()):
            if i % 2 == 0:
                new_ticks.append(np.round(tick))
                new_ticks_labels.append(str(int(np.round(tick))))

        cbar.set_ticks(new_ticks);
        cbar.set_ticklabels(new_ticks_labels);

        # override if vmin is provided, assume vmax is provided too for now
        if vmin is not None:
            cbar.set_ticks([vmin, (vmax+vmin)/2.0, vmax]);
            cbar.set_ticklabels(['%d'%(i) for i in [vmin, (vmax+vmin)/2.0, vmax]]);
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(label, fontsize=26, weight='bold')

    ax.axis('off')
    if std is not None:
        draw_circle_group(ax,100*np.array(mu),100*np.array(std))
    if frac:
        draw_circle_frac(ax,np.array(mu), color=color)
    else:
        draw_circle(ax,100*np.array(mu), color=color, fs=fs, xshift=xshift, yshift=yshift)
    if savepath is not None:
        if not cbar:
            plt.tight_layout()
        plt.savefig(savepath, dpi=600)
    plt.show()
    
def plot_bullseye_ratio(data,mu,vmin=None,vmax=None, savepath=None,cmap='RdBu_r', label='GPRS (%)', 
                  std=None,cbar=False,color='white',ptype='mesh',frac=False):
    
    rho     = np.arange(0,4,4.0/data.shape[1]);
    Theta   = np.deg2rad(range(data.shape[0]))
    [th, r] = np.meshgrid(Theta, rho);

    fig, ax = plt.subplots(figsize=(6,6))

    if ptype == 'mesh':
        im = ax.pcolormesh(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    else:
        im = ax.contourf(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    cbar = plt.colorbar(im, cax=fig.add_axes([0.15, -0.03, 0.7, 0.05]), orientation='horizontal')

    draw_circle_error(ax)
    ax.axis('off')
    if savepath is not None:
        if not cbar:
            plt.tight_layout()
        plt.savefig(savepath, dpi=600)
    plt.show()
    
def plot_bullseye_error(data,mu,vmin=None,vmax=None, savepath=None,cmap='RdBu_r', label='GPRS (%)',n=5):
    
    rho     = np.arange(0,4,4.0/data.shape[1]);
    Theta   = np.deg2rad(range(data.shape[0]))
    [th, r] = np.meshgrid(Theta, rho);

    fig, ax = plt.subplots(figsize=(6,6))

    levels = np.linspace(vmin, vmax, n+1)
    im = ax.contourf(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,levels=levels)

    cbar = plt.colorbar(im, cax=fig.add_axes([0.15, -0.03, 0.7, 0.05]), orientation='horizontal')

    #ticks = -np.array(range(0,120,20))

    #cbar.set_ticks(ticks);
    #cbar.set_ticklabels(['%d'%(i) for i in ticks]);

     

    ax.axis('off')
    draw_circle_error(ax)
    if savepath is not None:
        if not cbar:
            plt.tight_layout()
        plt.savefig(savepath, dpi=500)
    plt.show()

def draw_circle_error(ax,width=4):

    circle1 = plt.Circle((0,0), 1, color='black', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='black', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='black', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='black', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    j = 0
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)

    j += 6
    for i in range(4):
        theta_i = i*90*(np.pi/180) - 45
        xi, yi  = polar2cart(1, theta_i)
        xf, yf  = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)

def draw_circle_frac(ax, mu, width=4, fs=20, color='white'):
    
    
    
    circle1 = plt.Circle((0,0), 1, color='black', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='black', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='black', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='black', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    j = 0
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(3.5, theta_i + 2*np.pi/12)
        ax.text(xi-.3, yi, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color);
        xi, yi = polar2cart(2.5, theta_i + 2*np.pi/12)
        ax.text(xi-.3, yi, '%.2f' %(mu[j+6]), weight='bold', fontsize=fs, color=color); j += 1
        
    j += 6
    LABELS = ['ANT', 'SEPT', 'INF', 'LAT']
    for i in range(4):
        theta_i = i*90*np.pi/180 - 45 
        xi, yi = polar2cart(1, theta_i)
        xf, yf = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(1.5, theta_i + 2*np.pi/8)

        ax.text(xi-.3, yi, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color); j += 1;
        xi, yi = polar2cart(5, theta_i + 2*np.pi/8)

    ax.text(0-.3, 0-.3, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color)
    
def draw_circle(ax, mu, width=4, fs=15, xshift=0, yshift=0, color='white'):
    
    circle1 = plt.Circle((0,0), 1, color='black', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='black', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='black', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='black', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    j = 0
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(3.5, theta_i + 2*np.pi/12)
        ax.text(xi-.4-xshift, yi-yshift, '%.1f' %(mu[j]), weight='bold', fontsize=fs, color=color);
        xi, yi = polar2cart(2.5, theta_i + 2*np.pi/12)
        ax.text(xi-.4-xshift, yi-yshift, '%.1f' %(mu[j+6]), weight='bold', fontsize=fs, color=color); j += 1
        
    j += 6
    LABELS = ['ANT', 'SEPT', 'INF', 'LAT']
    for i in range(4):
        theta_i = i*90*np.pi/180  + 45*np.pi/180
        xi, yi = polar2cart(1, theta_i)
        xf, yf = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(1.5, theta_i + 2*np.pi/8)

        ax.text(xi-.4-xshift, yi-yshift, '%.1f' %(mu[j]), weight='bold', fontsize=fs, color=color); j += 1;
        xi, yi = polar2cart(5, theta_i + 2*np.pi/8)

    ax.text(-.4-xshift, 0-yshift, '%d' %(mu[j]), weight='bold', fontsize=fs, color=color)
  

def draw_circle_group(ax, mu, std, width=4, fs=14, color='white'):
    
        
    circle1 = plt.Circle((0,0), 1, color='black', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='black', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='black', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='black', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    j = 0
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(3.5, theta_i + 2*np.pi/12)
        ax.text(xi-.6, yi, '%d(%d)' %(mu[j],std[j]), weight='bold', fontsize=fs, color=color);
        xi, yi = polar2cart(2.5, theta_i + 2*np.pi/12)
        ax.text(xi-.6, yi, '%d(%d)' %(mu[j+6],std[j+6]), weight='bold', fontsize=fs, color=color); j += 1
        
    j += 6
    LABELS = ['ANT', 'SEPT', 'INF', 'LAT']
    for i in range(4):
        theta_i = i*90*np.pi/180 
        xi, yi = polar2cart(1, theta_i)
        xf, yf = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(1.5, theta_i + 2*np.pi/8)

        ax.text(xi-.6, yi-0.1, '%d(%d)' %(mu[j],std[j]), weight='bold', fontsize=fs, color=color); j += 1;
        xi, yi = polar2cart(5, theta_i + 2*np.pi/8)

    ax.text(0-.3, 0-.2, '%d(%d)' %(mu[j],std[j]), weight='bold', fontsize=fs, color=color)
    
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y