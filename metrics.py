import math
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from util import *

def density(game,segment):
    ds = densities[game]
    total = 0
    for l in segment:
        for d in ds:
            total += l.count(d)
    return total

def leniency(level):
    total = 0
    for l in level:
        total += len(l)-l.count('E')-l.count('H')-l.count('t')-l.count('C') #-l.count('e')-l.count('v')-l.count('^')-l.count('ě')-l.count('Ě')-l.count('ѷ')-l.count('⌅')
    return total

def difficulty(segment):
    total = 0
    for i, l in enumerate(segment):
        total += l.count('E')+l.count('H')+l.count('t')+l.count('C')
        #if GAME == 'smb':
        if i == 15:
            total += (0.5)*l.count(' ') # 0.5 penalty for gap in last row
    return int(round(total))

def interestingness(level):
    total = 0
    for l in level:
        total += l.count('o') + l.count('Q') + l.count('?') + l.count('D') + l.count('*') + l.count('+') + l.count('L') + l.count('U') + l.count('W') + l.count('l') + l.count('W')
    return total

def nonlinearity(segment,game='smb'):
    if game != 'ki':
        level = [[segment[j][i] for j in range(len(segment))] for i in range(len(segment[0]))]
    else:
        level = segment
    x = np.arange(16)
    y = []
    for i, lev in enumerate(level):
        appended = False
        for j, l in enumerate(lev):
            if l != '-' and l != 'P':
                y.append(15-j)
                appended = True
                break
        if not appended:
            y.append(0)
    x = x.reshape(-1,1)
    y = np.asarray(y)
    reg = linear_model.LinearRegression()
    reg.fit(x,y)
    y_pred = reg.predict(x)
    mse = mean_squared_error(y,y_pred)
    return (int(round(mse)))
    #return mse/53 # ALL
    #if GAME == 'smb':
    #    return (mse/26.87)  # SMB
    #elif GAME == 'ki':
    #    return (mse/57.93)   # KI
    #elif GAME == 'mm':
    #    return (mse/70.09)  # MM
    #return mse

def h_symmetry(level):
    total = 0
    for l in level:
        l1, l2 = l[:8], l[8:]
        for a,b in zip(l1,l2):
            if a == b: #and a != '-' and a != 'P' and b != '-' and b != 'P':
                total += 1
    return total

def v_symmetry(level):
    level_no_mid = level[:len(level)//2] + level[(len(level)//2)+1:]
    level_t = [[level_no_mid[j][i] for j in range(len(level_no_mid))] for i in range(len(level_no_mid[0]))]
    return h_symmetry(level_t)

def symmetry(level):
    #return (h_symmetry(level)+v_symmetry(level))/256
    return h_symmetry(level)+v_symmetry(level)

def h_similarity(level,level_rows):
    total = 0
    for l in level:
        l_str = ''.join(list(l))
        if l_str in level_rows:
            total += 1
    return total

def v_similarity(level,level_cols):
    total = 0
    level_t = [[level[j][i] for j in range(len(level))] for i in range(len(level[0]))]
    for l in level_t:
        l_str = ''.join(list(l))
        if l_str in level_cols:
            total += 1
    return total

def similarity(level,level_rows,level_cols):
    #return (h_similarity(level)+v_similarity(level))/32
    return h_similarity(level,level_rows)+v_similarity(level,level_cols)

def h_dissimilarity(level,level_rows):
    total = 0
    for l in level:
        l_str = ''.join(list(l))
        if l_str not in level_rows:
            total += 1
    return total

def v_dissimilarity(level,level_cols):
    total = 0
    level_t = [[level[j][i] for j in range(len(level))] for i in range(len(level[0]))]
    for l in level_t:
        l_str = ''.join(list(l))
        if l_str not in level_cols:
            total += 1
    return total

def dissimilarity(level,level_rows,level_cols):
    #return (h_dissimilarity(level)+v_dissimilarity(level))/32
    return h_dissimilarity(level,level_rows)+v_dissimilarity(level,level_cols)

def v_traversability(level):
    total = 0
    for i, (l1,l2) in enumerate(zip(level[:-1],level[1:])):
        #print('i', i)
        l1_str = ''.join(list(l1))
        l2_str = ''.join(list(l2))
        #print(l1_str, '\t', l2_str, '\t')
        if 'P' not in l1_str or 'P' not in l2_str:
            total += 0
            continue
        #delta = 16 - math.fabs(l1_str.find('P') - l2_str.find('P'))
        delta = int(math.fabs(l1_str.find('P') - l2_str.find('P')))
        #print(delta)
        if delta < 2:
            total += 1
        
        #total += delta
    #print('tot', total)
    return (total/15)

def h_traversability(level):
    level_t = [[level[j][i] for j in range(len(level))] for i in range(len(level[0]))]
    return v_traversability(level_t)

def traversability(level):
    return max(h_traversability(level), v_traversability(level))

"""
def optimize_traversability(z):
    z = z.reshape(1,-1)
    z = torch.DoubleTensor(z)
    #z = z.to(dtype=torch.float64)
    z = z.to(device)
    #print(z.shape)
    level = get_segment_from_z(z)
    #z_decoded = model.decode(z)
    #level = z_decoded.data.cpu().numpy()
    #level = z_decoded.view(1, 134, 6)
    #vs = 75
    #return calc_voidsize(level[0], vs)
    return (1.0 - traversability(level))
"""