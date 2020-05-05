boundaries=660,1429

def get_half_cords(width,height):
    
    if width%2==0:
        w1,w2=width//2,width//2
    if width%2==1:
        w1,w2=width//2 +1 ,width//2

    if height%2==0:
        h1,h2=height//2,height//2

    if height%2==1:
        h1,h2=height//2 +1 ,height//2

    return w1,w2,h1,h2

def car_on_corners(im,x,y):
    
    
    
    width,height=im.size
    
    w1,w2,h1,h2=get_half_cords(width,height)
    
    
    left,top,right,bottom=0,0,width,height
    
    if w2+x>boundaries[0]:
        w2_diff=w2+x-boundaries[0]
        right=width-w2_diff
        
    if x-w1<0:
        w1_diff=abs(x-w1)
        left=w1_diff
        
    
    
    if h2+y>boundaries[1]:
        
        h2_diff=h2+y-boundaries[1]
        # top=h2_diff
        bottom=height-h2_diff
    
    if y-h1<0:
        h1_diff=abs(y-h1)
        # bottom=height-h1_diff
        top=h1_diff
        
    return left,top,right,bottom


def get_car_paste_cordinates(im,x,y):
    
    
    width,height=im.size
    
    w1,w2,h1,h2=get_half_cords(width,height)
 
    return max(0,x-w1),max(y-h1,0),min(x+w2,boundaries[0]) ,min(y+h2,boundaries[1])


def get_sand_crop_coordinates(x,y,no):
    

    x_coordinates=(x-no,x+no)
    y_coordinates=(y-no,y+no)
    
    if x+no>boundaries[0]:
        x_coordinates=(boundaries[0] - 2*no,boundaries[0])
    
    if x-no<0:
        x_coordinates=(0,2*no)

    if y+no>boundaries[1]:
        y_coordinates=(boundaries[1] - 2*no,boundaries[1])

    if y-no<0:
        y_coordinates=(0,2*no)
        
    return x_coordinates[0],y_coordinates[0],x_coordinates[1],y_coordinates[1] 


