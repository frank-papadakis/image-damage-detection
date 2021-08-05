

### LIBRARIES
import copy
import numpy as np
from numpy.random import default_rng
### SEEDS ###

rng=default_rng()

class damage():
  def square(array,number_of_squares,damage_radius):
    shape=np.array(np.shape(array))
    artifact=np.zeros(shape=[shape[0],shape[1],shape[2]])
    final=np.zeros(shape=[shape[0],shape[1],shape[2]])
    for i in range(number_of_squares):
      center=np.c_[np.random.randint(damage_radius,shape[1]-damage_radius+1,size=shape[0]).tolist(),np.random.randint(damage_radius,shape[2]-damage_radius+1,size=shape[0]).tolist()]
      for i in range(shape[0]):
        artifact[i,center[i,0]-damage_radius:center[i,0]+damage_radius,center[i,1]-damage_radius:center[i,1]+damage_radius]=np.random.random()*2-1
    damage=np.where(artifact!=0,artifact,array).reshape(-1,28,28)
    mask=copy.copy(artifact)
    mask[np.where(artifact!=0)]=1
    mask=mask.reshape(-1,784)
    return damage, mask

  def lines(array, horizontal_number, vertical_number):              
    shape=np.array(np.shape(array))
    artifact=np.zeros(shape=shape)
    for i in range(shape[0]):
      horizontal=rng.choice(shape[1],size=horizontal_number,replace=False)
      vertical=rng.choice(shape[2],size=vertical_number,replace=False)
      artifact[i,:,horizontal]=np.random.random()*2-1    
      artifact[i,vertical,:]=np.random.random()*2-1
    damage=np.where(artifact!=0,artifact,array).reshape(-1,28,28)
    mask=copy.copy(artifact)
    mask[np.where(artifact!=0)]=1
    mask=mask.reshape(-1,784)
    return damage, mask

  def noise(array, intensity):
    shape=np.array(np.shape(array))
    length=shape[1]*shape[2]
    mask_len=int(intensity*length)
    artifact=np.zeros(shape=[shape[0],mask_len]).astype(int)
    damage=copy.copy(array).reshape(-1,length)
    mask=np.zeros(shape=[shape[0],length]).astype(int)
    for i in range(shape[0]):
      artifact[i]=rng.choice(length,mask_len,replace=False).astype(int)
      damage[i,artifact[i]]=np.random.random()*2-1
      mask[i,artifact[i]]=1
    damage=damage.reshape(-1,shape[1],shape[2])
    return damage,mask


def create_damage(images, amount):
    images = images.reshape((-1, 28, 28))
    if amount=='low':
        square_large_damage,square_large_mask=damage.square(images,1,5)
        squares_small_damage,squares_small_mask=damage.square(images,5,3)
        vertical_damage,vertical_mask=damage.lines(images,8,0)
        horizontal_damage,horizontal_mask=damage.lines(images,0,8)
        perpendicular_damage,perpendicular_mask=damage.lines(images,5,5)
        noise_damage, noise_mask=damage.noise(images,0.2)
    
    
    if amount=='medium':
        square_large_damage,square_large_mask=damage.square(images,1,6)
        squares_small_damage,squares_small_mask=damage.square(images,5,3)
        vertical_damage,vertical_mask=damage.lines(images,12,0)
        horizontal_damage,horizontal_mask=damage.lines(images,0,12)
        perpendicular_damage,perpendicular_mask=damage.lines(images,8,8)
        noise_damage, noise_mask=damage.noise(images,0.3)
        
    if amount=='high':
        square_large_damage,square_large_mask=damage.square(images,1,6)
        squares_small_damage,squares_small_mask=damage.square(images,5,3)
        vertical_damage,vertical_mask=damage.lines(images,14,0)
        horizontal_damage,horizontal_mask=damage.lines(images,0,14)
        perpendicular_damage,perpendicular_mask=damage.lines(images,8,8)
        noise_damage, noise_mask=damage.noise(images,0.4)
    length=np.array(np.shape(images))    
    control, control_mask = images.reshape(-1,28,28), np.zeros(shape=[length[0],784])    
    damaged=np.concatenate((horizontal_damage,vertical_damage,square_large_damage,squares_small_damage,perpendicular_damage,noise_damage,control)) # ,vertical_damage,horizontal_damage, 
    mask=np.concatenate((horizontal_mask,vertical_mask,square_large_mask,squares_small_mask,perpendicular_mask,noise_mask,control_mask)) 
    
    return damaged, mask  
        