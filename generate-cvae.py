import argparse, random, torch, os, math, json, sys, re
import torch.optim as optim
import torch.utils.data
import numpy as np
from PIL import Image
from model_lin_cond import get_cond_model, load_cond_model
import pickle

device = torch.device('cuda')
latent_dim = 8 # size of latent vector
GAME = 'blend_zelda_lode'  # smb, ki, mm, smb_pats, all

label_sizes = {'dg':4,'zelda':4,'met':4,'mm':4,'lode':4,'blend_dz':16, 'blend_dz_2': 2, 'blend_dz_common':2, 
               'blend_dz_common_full':16, 'blend_dz_common_doors':6, 'blend_met_mm':6, 'blend_zelda_lode':6}
label_size = label_sizes[GAME]
num_labels = int(math.pow(2,label_size))

dims = (15,16) if 'met' in GAME or 'mm' in GAME else (11,16)

zelda_folder = 'zelda_rooms/'
dg_folder = 'dg_chunks_edited/'
met_folder = 'met_chunks_all/'
lode_folder = 'lode_chunks/'
mm_folder = 'mm_chunks_all/'
""" 
1
0.007474425973343034 0.19019087619696037
-0.02488125679941893 0.16369227385271515

1
0.010828688050717575 0.17511510800256228
-0.019412209588609192 0.17028894044282658

10
0.00771665663011336 0.18336345620910036
-0.018981274527597205 0.1719623821683873

100
0.0067251006391976596 0.18052977787175753
-0.015101091382368361 0.170988833379435

1000
0.007690180346566041 0.1813693650824676
-0.015203464839032359 0.17129609054447595

"""

dg_mean = 0.01 #0.007690180346566041 #-0.013997895605728334
dg_std = 0.18 #0.1813693650824676 #0.17229445236817564
zelda_mean = -0.02 #-0.015203464839032359 #0.01327264206087219
zelda_std = 0.17 #0.17129609054447595 #0.1696900470971206 

folders = {'dg':dg_folder,'zelda':zelda_folder,'blend_dz':None, 'blend_dz_2':None, 'blend_dz_common':None,
           'blend_dz_common_full':None, 'met':met_folder, 'lode':lode_folder, 'mm':mm_folder}
if not GAME.startswith('blend'):
    folder = folders[GAME]
#manual_seed = random.randint(1, 10000)
#random.seed(manual_seed)
#torch.manual_seed(0)
#np.random.seed(0)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    
def parse_folder(folder,g,dir=False):
    levels, text, dirs = [], '', []
    files = os.listdir(folder)
    files[:] = (value for value in files if value != '.')
    files = natural_sort(files)
    for file in files:
        if file.startswith('.'):
            continue
        if dir:
            d1 = file[file.rfind('_')+1:]
            d2 = d1[:d1.find('.')]
            dirs.append(d2)
        with open(os.path.join(folder,file),'r') as infile:
            level = []
            for line in infile:
                #if g.startswith('blend'):
                #    line = line.replace('F','-')
                #    line = line.replace('W','X')
                if g == 'lode' and GAME == 'blend_zelda_lode':  # disambiguate B and M in Lode
                    line = line.replace('B','#')
                    line = line.replace('M','N')  
                elif g == 'mm':
                    line = line.replace('P','-')
                elif g == 'met':
                    line = line.replace(')','-')
                    line = line.replace('(','-')
                    line = line.replace('v','-')
                    line = line.replace('[','#')  
                    line = line.replace(']','#')
                text += line
                level.append(list(line.rstrip()))
            levels.append(level)
    return levels, text, dirs

"""
if not GAME.startswith('blend'):
    levels, text = parse_folder(folder)
    text = text.replace('\n','')
    print(len(levels))
else:
    dg_levels, dg_text = parse_folder(dg_folder)
    dg_text = dg_text.replace('\n','')
    zelda_levels, zelda_text = parse_folder(zelda_folder)
    zelda_text = zelda_text.replace('\n','')
    #zelda_upsampled = zelda_levels[:]
    #while len(zelda_upsampled) < len(dg_levels):
    #    zelda_upsampled.append(random.choice(zelda_levels))
    text = dg_text + zelda_text
    text = text.replace('\n','')
"""
dir = True if GAME != 'zelda' else False
if not GAME.startswith('blend'):
    levels, text, dirs = parse_folder(folder,GAME,dir)
    text = text.replace('\n','')
    print(len(levels),len(dirs))
else:
    if GAME == 'blend_met_mm':
        met_levels, met_text, met_dirs = parse_folder(met_folder,'met',dir)
        mm_levels, mm_text, mm_dirs = parse_folder(mm_folder,'mm',dir)
        met_text = met_text.replace('\n','')
        mm_text = mm_text.replace('\n','')
        text = met_text + mm_text
        text = text.replace('\n','')
    elif GAME == 'blend_zelda_lode':
        zelda_levels, zelda_text, _ = parse_folder(zelda_folder,'zelda')
        zelda_text = zelda_text.replace('\n','')
        lode_levels, lode_text, lode_dirs = parse_folder(lode_folder,'lode',dir)
        lode_text = lode_text.replace('\n','')
        text = zelda_text + lode_text
        text = text.replace('\n','')
    
chars = sorted(list(set(text.strip('\n'))))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
print(char2int)
num_tiles = len(char2int)
print(num_tiles)
#sys.exit()
def get_label_dg(room):
    # {'#': 0, '*': 1, '-': 2, 'X': 3, '^': 4}
    label = [False] * 4 # #, *, X, ^
    room_string = ''
    for l in room:
        room_string += ''.join(l)
    if '#' in room_string:
        label[0] = True
    if '*' in room_string:
        label[1] = True
    if 'X' in room_string:
        label[2] = True
    if '^' in room_string:
        label[3] = True
    return label

def get_door_label(room):
    label = [False] * 4
    room_string = ''
    for l in room:
        room_string += ''.join(l)
    room_t = [''.join(s) for s in zip(*room)]
    if 'D' in room[1]:
        label[0] = True
    if 'D' in room[len(room)-2]:
        label[1] = True
    if 'D' in room_t[1]:
        label[2] = True
    if 'D' in room_t[len(room_t)-2]:
        label[3] = True
    return label

def get_label_zelda(room):
    label = [False] * 10   # D (N/S/W/E), M, B, I, S, O, P
    room_string = ''
    for l in room:
        room_string += ''.join(l)
    room_t = [''.join(s) for s in zip(*room)]
    if 'D' in room[1]:
        label[0] = True
    if 'D' in room[len(room)-2]:
        label[1] = True
    if 'D' in room_t[1]:
        label[2] = True
    if 'D' in room_t[len(room_t)-2]:
        label[3] = True
    if 'M' in room_string:
        label[4] = True
    if 'B' in room_string:
        label[5] = True
    if 'I' in room_string:
        label[6] = True
    if 'S' in room_string:
        label[7] = True
    if 'O' in room_string:
        label[8] = True
    if 'P' in room_string:
        label[9] = True
    return label


def get_label_dir(dir):
    label = [False] * 4 # U, D, L, R
    if 'U' in dir:
        label[0] = True
    if 'D' in dir:
        label[1] = True
    if 'L' in dir:
        label[2] = True
    if 'R' in dir:
        label[3] = True
    return label

def get_label_mm_3(dir):
    label = [False] * 3 # U, D, R
    if 'U' in dir:
        label[0] = True
    if 'D' in dir:
        label[1] = True
    if 'R' in dir:
        label[2] = True
    return label

def get_label_met(chunk):
    label = [False] * 5 # +, B, D, E, ^
    chunk_string = ''
    for l in chunk:
        chunk_string += ''.join(l)
    if '+' in chunk_string:
        label[0] = True
    if 'B' in chunk_string:
        label[1] = True
    if 'D' in chunk_string:
        label[2] = True
    if 'E' in chunk_string:
        label[3] = True
    if '^' in chunk_string:
        label[4] = True
    return label

def get_label_blend(level):
    dg_label = get_label_dg(level)
    zd_label = get_label_zelda(level)
    return dg_label + zd_label

def get_segment_from_zc(z,c):
    level = model.decoder.decode(z,c)
    level = level.reshape(level.size(0),num_tiles,dims[0],dims[1])
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1).squeeze(0)
    level = np.zeros(im.shape)
    level = []
    for i in im:
        level.append(''.join([int2char[t] for t in i]))
    return level

def get_z_from_segment(segment):
    out = []
    for l in segment:
        l = list(l)
        l_map = [char2int[x] for x in l]
        out.append(l_map)
    out = np.asarray(out)
    out_onehot = np.eye(num_tiles, dtype='uint8')[out]
    out_onehot = np.rollaxis(out_onehot, 2, 0)
    out_onehot = out_onehot[None, :, :]
    out = torch.DoubleTensor(out_onehot)
    out = out.to(device)
    out_lin = out.view(out.size(0),-1)
    z, _, _ = model.encoder.encode(out_lin)
    return z

def get_segment_from_z(z):
    level = model.decoder.decode(z)
    level = level.reshape(level.size(0),num_tiles,dims[0],dims[1])
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1).squeeze(0)
    level = np.zeros(im.shape)
    level = []
    for i in im:
        level.append(''.join([int2char[t] for t in i]))
    return level

def get_label_tensor(label):
    return torch.DoubleTensor(label).reshape(1,-1).to(device)

get_label_func = {'dg':get_label_dg, 'zelda':get_label_zelda, 'blend_dz':get_label_blend, 'blend_dz_2': None, 
                  'blend_dz_common_full':get_label_blend, 'blend_dz_common_doors':get_label_blend,'met':get_label_met,
                  'lode':None,'mm':None,'blend_met_mm':None,'blend_zelda_lode':None}
if not GAME.startswith('blend'):
    get_label = get_label_func[GAME]
input_dim = 240 if 'met' in GAME or 'mm' in GAME else 176
model_name = 'cvae_lean_' + GAME + '_ld_' + str(latent_dim) + '_final.pth'
#vae_model_name = 'vae_' + GAME + '_ld_' + str(latent_dim) + '_final.pth'
model = load_cond_model(model_name,input_dim,num_tiles,latent_dim,label_size,device)
#vae = load_model(vae_model_name,176,num_tiles,num_tiles,latent_dim,device)
model.eval()
model.to(device)
#print(model)
#vae.eval()
#vae.to(device)
#sys.exit()

"""
z_dg, z_zelda = [], []
ovr_dg, ovr_zelda = [], []
ovr_dg_std, ovr_zelda_std = [], []
for i in range(1000):
    print(i)
    for level in dg_levels:
        z = get_z_from_segment(level)
        z_dg.append(z.mean().item())
    for level in zelda_levels:
        z = get_z_from_segment(level)
        z_zelda.append(z.mean().item())
    ovr_dg.append(np.mean(z_dg))
    ovr_zelda.append(np.mean(z_zelda))
    ovr_dg_std.append(np.std(z_dg))
    ovr_zelda_std.append(np.std(z_zelda))
print(np.mean(ovr_dg), np.mean(ovr_dg_std))
print(np.mean(ovr_zelda), np.mean(ovr_zelda_std))
sys.exit()
"""
"""
for blend in [1, 0.75, 0.5, 0.25, 0]:
    print(blend, '\t', (1-blend))
    blend_mean = blend*dg_mean + ((1.0-blend)*zelda_mean)
    blend_std = blend*dg_std + ((1.0-blend)*zelda_std)
    z = torch.DoubleTensor(1,latent_dim).normal_(blend_mean,blend_std).to(device)
    segment = get_segment_from_z(z)
    print('\n'.join(segment))
sys.exit()
"""
if GAME.startswith('blend'):
    met_label, mm_label, bmm_label, none_label = [1,0], [0,1], [1,1], [0,0]
    z = torch.DoubleTensor(1,latent_dim).normal_(0,1).to(device)
    for i in range(16):
        dirs = bin(i)[2:].zfill(4)
        dir_label = [int(x) for x in dirs]
        full_met = met_label + dir_label
        full_mm = mm_label + dir_label
        full_bmm = bmm_label + dir_label
        full_none = none_label + dir_label
        for l in [full_met, full_mm, full_bmm, full_none]:
            print('Label: ', l)
            label_tensor = get_label_tensor(l)
            segment = get_segment_from_zc(z,label_tensor)
            print('\n'.join(segment),'\n')
sys.exit()
content_label = [1,1,1,0,1,0,0,0,0,1]
zelda_label = [1,0]
dg_label = [0,1]
blend_label = [1,1]
door_label = [0,0,0,1]

content = [0,0,0,1,0]
totals = []
z = torch.DoubleTensor(1,latent_dim).normal_(0,1).to(device)
for j in range(1):
    #print(j)
    cont = bin(j)[2:].zfill(5)
    cont = [int(x) for x in cont]
    cont = [1,0,0,0,0]
    total = 0
    for i in range(16):
        doors = bin(i)[2:].zfill(label_size)
        doors = [int(x) for x in doors]
        label = doors #+ cont
        print(label)
        label_tensor = get_label_tensor(label)
        segment = get_segment_from_zc(z,label_tensor)
        print('\n'.join(segment),'\n')
        continue
        if GAME == 'zelda':
            door_label = get_door_label(segment)
            door_label = np.array(door_label).astype('uint8').tolist()
            if doors == door_label:
                total += 1
            else:
                print(doors,'\t',door_label)
                print('\n'.join(segment))
        elif GAME == 'met':
            cont_label = get_label_met(segment)
            cont_label = np.array(cont_label).astype('uint8').tolist()
            #print(cont, ' ', cont_label)
            if cont == cont_label:
                total += 1
    totals.append(total)

#print(totals)
print(np.mean(totals))
sys.exit()
z = torch.DoubleTensor(1,latent_dim).normal_(0,1).to(device)
for game_label in [zelda_label,dg_label,blend_label]:
    label = game_label + door_label
    label_tensor = get_label_tensor(label)
    print(label)
    segment = get_segment_from_zc(z,label_tensor)
    print('\n'.join(segment),'\n')
sys.exit()
    
if GAME.startswith('blend'):
    zelda_label = [1,0]
    dg_label = [0,1]
    blend_label = [1,1]

    for label in [zelda_label,dg_label,blend_label]:
        print(label)
        label_tensor = get_label_tensor(label)
        segment = get_segment_from_zc(z,label_tensor)
        print('\n'.join(segment),'\n')

    sys.exit()

    
#label_tensor = get_label_tensor(label)
#segment = get_segment_from_zc(z,label_tensor)
#print('\n'.join(segment))
#sys.exit()
if GAME == 'dg':
    for l in range(num_labels):
        label = [int(j) for j in bin(l)[2:]]
        if len(label) < label_size:
            label = [0] * (label_size - len(label)) + label
        print(label)
        label_tensor = get_label_tensor(label)
        segment = get_segment_from_zc(z,label_tensor)
        print('\n'.join(segment))
else:
    for i in range(6):
        break
        label = [0] * 6
        label[i] = 1
        label = [0,1,0,0] + label
        print(label)
        label_tensor = get_label_tensor(label)
        segment = get_segment_from_zc(z,label_tensor)
        print('\n'.join(segment))