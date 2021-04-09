from __future__ import print_function
import argparse, random, torch, os, math, json, sys, re, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model_lin_cond import get_cond_model, load_cond_model
from torch.utils.data import DataLoader, Dataset, TensorDataset
from util import *
from tile_images import *

parser = argparse.ArgumentParser(description='Generate room layout.')
parser.add_argument('--no-door-ns', action='store_true', help='Disallow NS doors. (default: false)')
parser.add_argument('--no-door-ew', action='store_true', help='Disallow EW doors. (default: false)')
parser.add_argument('--no-open-ns', action='store_true', help='Disallow NS open. (default: false)')
parser.add_argument('--no-open-ew', action='store_true', help='Disallow EW open. (default: false)')
parser.add_argument('--no-loops', action='store_true', help='Disallow loops. (default: false)')
parser.add_argument('--door-pct', type=int, help='Percent chance of door when there is a choice. (default: %(default)s)', default=20)
parser.add_argument('--min-rooms', type=int, help='Minimum rooms. (default: %(default)s)', default=5)
parser.add_argument('--max-rooms', type=int, help='Maximum rooms.(default: %(default)s)', default=20)
parser.add_argument('--json', action='store_true', help='Output JSON. (default: false)')
parser.add_argument('--game', type=str, help='Game (options: met, mm, lode, zelda, blend_met_mm, blend_zelda_lode)', default='met')
parser.add_argument('--ld', type=int, help='Latent dimension size', default=8)
parser.add_argument('--multi', action='store_true', help='Use multiple models (default: false)')
parser.add_argument('--met', type=float, help='met prob', default=0.33)
parser.add_argument('--zel', type=float, help='met prob', default=0.33)
parser.add_argument('--mm', type=float, help='met prob', default=0.33)
args = parser.parse_args()

device = torch.device('cpu')
GAME = args.game
latent_dim = args.ld

dims = (15,16) if 'mm' in GAME or 'met' in GAME else (11,16)

zelda_folder = 'zelda_rooms_new/'
met_folder = 'met_chunks_all/'
lode_folder = 'lode_chunks/'
mm_folder = 'mm_chunks_all/'

folders = {'zelda':zelda_folder, 'met':met_folder, 'lode':lode_folder, 'mm':mm_folder}
out_folders = {'zelda':'out_zelda/','met':'out_met/','lode':'out_lode/','mm':'out_mm/'}
all_images = {'met':met_images, 'lode':lode_images, 'zelda':zelda_images, 'mm':mm_images, 'blend_met_mm':bmm_images, 'blend_zelda_lode':bzl_images}
images = all_images[GAME]
folder = folders[GAME] if 'blend' not in GAME else None
label_size = 4 if 'blend' not in GAME else 6

NORTH  = 'north'
SOUTH  = 'south'
EAST   = 'east'
WEST   = 'west'
DIRS   = [NORTH, SOUTH, EAST, WEST]

CLOSED = 'closed'
DOOR   = 'door'
OPEN   = 'open'

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

def get_segment_from_zc(game_model,z,c,game=GAME):
    dim = (15,16) if 'mm' in game or 'met' in game else (11,16)
    level = game_model.decoder.decode(z,c)
    level = level.reshape(level.size(0),len(char2ints[game]),dim[0],dim[1])
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1).squeeze(0)
    level = np.zeros(im.shape)
    level = []
    for i in im:
        level.append(''.join([int2chars[game][t] for t in i]))
    return level

def get_label_tensor(label):
    return torch.DoubleTensor(label).reshape(1,-1).to(device)

def sample_z():
    return torch.DoubleTensor(1,latent_dim).normal_(0,1).to(device)

def get_image_from_segment(segment,game=GAME):
    images = all_images[game]
    if args.multi:
        dim = (15,16)
    else:
        dim = (15,16) if 'mm' in game or 'met' in game else (11,16)
    img = Image.new('RGB',(dim[1]*16, dim[0]*16))
    for row, seq in enumerate(segment):
        for col, tile in enumerate(seq):
            img.paste(images[tile],(col*16,row*16))
    return img

dir = True if GAME != 'zelda' else False
if not GAME.startswith('blend'):
    levels, text, dirs = parse_folder(folder,GAME,dir)
    text = text.replace('\n','')
    print(len(levels),len(dirs))

"""
def pad_zelda(segment):
    padded = []
    padded.append('W' * 16)  # outer north wall
    #padded.append('W' * 16)  # wall pad
    padded.append(segment[1])  # inner north wall of original segment that may/may not have doors
    padded.append(segment[2])  # outer floor area north pad
    padded.append(segment[2])  # outer floor area north pad
    padded.extend(segment[2:-2])  # rest of segment
    padded.append(segment[-3])  # outer floor area south pad
    padded.append(segment[-3])  # outer floor area south pad
    padded.append(segment[-2])  # inner south wall of original segment that may/may not have doors
    #padded.append('W' * 16)  # wall pad
    padded.append('W' * 16)  # outer south wall
    print(len(padded))
    print('\n'.join(padded))
    print('\n','\n'.join(segment))
    return padded
"""
#chars = sorted(list(set(text.strip('\n'))))
#int2char = dict(enumerate(chars))
#char2int = {ch: ii for ii, ch in int2char.items()}
char2int, int2chars = char2ints[GAME], {}
for g in ['met','mm','zelda','lode']:
    c2i = char2ints[g]
    i2c = {ch: ii for ii, ch in c2i.items()}
    int2chars[g] = i2c
#int2char = {ch: ii for ii, ch in char2int.items()}
int2char = int2chars[GAME]
num_tiles = len(char2int)
#num_tiles = len(char2ints[GAME])
print(num_tiles)

input_dim = 240 if 'met' in GAME or 'mm' in GAME else 176
model_name = 'models/cvae_lean_' + GAME + '_ld_' + str(latent_dim) + '_final.pth'
model = load_cond_model(model_name,input_dim,num_tiles,latent_dim,label_size,device)
model.eval()
model.to(device)

models = {}
if args.multi:
    for game in ['met','mm','zelda','lode']:
        input_dim = 240 if 'met' in game or 'mm' in game else 176
        num_tiles = len(char2ints[game])
        model_name = 'models/cvae_lean_' + game + '_ld_' + str(latent_dim) + '_final.pth'
        game_model = load_cond_model(model_name,input_dim,num_tiles,latent_dim,label_size,device)
        game_model.eval()
        game_model.to(device)
        models[game] = game_model

layout = {}
layout[(0, 0)] = {}
layout[(0, 0)][NORTH] = CLOSED
layout[(0, 0)][SOUTH] = CLOSED
layout[(0, 0)][EAST]  = CLOSED
layout[(0, 0)][WEST]  = CLOSED

def neighbor(cell, dr):
    if dr == NORTH:
        return (cell[0], cell[1] - 1)
    elif dr == SOUTH:
        return (cell[0], cell[1] + 1)
    elif dr == EAST:
        return (cell[0] + 1, cell[1])
    elif dr == WEST:
        return (cell[0] - 1, cell[1])
    else:
        raise RuntimeError('invalid direction')

def opposite(dr):
    if dr == NORTH:
        return SOUTH
    elif dr == SOUTH:
        return NORTH
    elif dr == EAST:
        return WEST
    elif dr == WEST:
        return EAST
    else:
        raise RuntimeError('invalid direction')

if args.no_door_ns and args.no_open_ns and args.no_door_ew and args.no_open_ew:
    raise RuntimeError('no way to connect')

rooms = random.randint(args.min_rooms, args.max_rooms)
if GAME == 'mm':
    segments = random.randint(10,15)
    prev = None
    if random.random() < 0.75:
        dirs = ['right']
    else:
        dirs = ['up']
    for i in range(1,segments):
        prev = dirs[i-1]
        r = random.random()
        if prev == 'up':
            next = 'up' if r < 0.7 else 'right'
        elif prev == 'right':
            if r < 0.8:
                next = 'right'
            elif r < 0.9:
                next = 'up'
            else:
                next = 'down'
        elif prev == 'down':
            next = 'down' if r < 0.7 else 'right'
        dirs.append(next)
    print(dirs)
    
    layout, x, y = {}, 0, 0
    for i, dir in enumerate(dirs):
        label = [0] * 4
        if i == 0:
            label = [0,0,0,1] if dir == 'right' else [1,0,0,0]
            layout[(x,y)] = (label, dir)
        elif i == len(dirs)-1: # last
            prev = dirs[i-1]
            if prev == 'right':
                label[2] = 1  # left
            elif prev == 'up':
                label[1] = 1 # down
            elif prev == 'down':
                label[0] = 1 # up

            if dir == 'right':
                x += 1
            elif dir == 'up':
                y -= 1
            elif dir == 'down':
                y += 1
            layout[(x,y)] = (label, dir)
        else:
            prev, next = dirs[i-1], dirs[i+1]
            if prev == 'right':
                label[2] = 1  # left
            elif prev == 'up':
                label[1] = 1 # down
            elif prev == 'down':
                label[0] = 1 # up
            
            if next == 'right':
                label[3] = 1
            elif next == 'up':
                label[0] = 1
            elif next == 'down':
                label[1] = 1
            
            if dir == 'right':
                x += 1
            elif dir == 'up':
                y -= 1
            elif dir == 'down':
                y += 1
            layout[(x,y)] = (label, dir)
        print(dir, label, (x,y))
    
    cells = list(layout.keys())
    x_lo = min(cells)[0]
    x_hi = max(cells)[0]
    y_lo = min(cells, key=lambda x: x[1])[1]
    y_hi = max(cells, key=lambda x: x[1])[1]
    width, height = abs(x_lo - x_hi)+1, abs(y_hi - y_lo)+1
    x_adj, y_adj = abs(x_lo * 256 - 0), abs((y_lo * dims[0] * dims[1]) - 0)
    layout_img = Image.new('RGB',(width*256, height*(dims[0]*dims[1])))
    for key in layout:
        z = sample_z()
        label, dir = layout[key]
        label = get_label_tensor(label)
        segment = get_segment_from_zc(model,z,label)
        img = get_image_from_segment(segment)
        x, y = key
        layout_img.paste(img, ((x*256)+x_adj,(y*dims[0]*dims[1])+y_adj))
        print(x, y, '\t', (x*256)+x_adj, (y*dims[0]*dims[1])+y_adj, '\t', label, '\t', dir)
        print('\n'.join(segment),'\n')
    layout_img.save('layout_' + GAME + '_' + str(latent_dim) + '.png')
    sys.exit()
elif GAME == 'lode':
    rows, cols = random.randint(2,4), random.randint(2,4)
    layout_img = Image.new('RGB',(cols*256, rows*(dims[0]*dims[1])))
    for y in range(rows):
        for x in range(cols):
            label = [0] * 4
            if y == 0: # top row
                label[1] = 1  # down set to 1
            elif y == rows-1:  # bottom row
                label[0], label[1] = 1, 0  # up set to 1, down set to 0
            
            if x == 0:  # first column
                label[2], label[3] = 0, 1  # left set to 0, right set to 1
            elif x == cols-1:
                label[2], label[3] = 1, 0  # reverse of previous
            
            if x !=0 and x != cols-1 and y != 0 and y != rows-1:
                label = [1,1,1,1]  # non perimeter cell
            z = sample_z()
            label = get_label_tensor(label)
            segment = get_segment_from_zc(model,z,label)
            img = get_image_from_segment(segment)
            layout_img.paste(img, ((x*256),(y*dims[0]*dims[1])))
            print(x, y, '\t', (x*256), (y*dims[0]*dims[1]), '\t', label)
            print('\n'.join(segment),'\n')
    layout_img.save('layout_' + GAME + '_' + str(latent_dim) + '.png')
    sys.exit()
        
            



        
for ii in range(rooms):
    cells = list(layout.keys())

    options = []
    for cell in cells:
        for dr in DIRS:
            if dr in [NORTH, SOUTH] and args.no_door_ns and args.no_open_ns:
                continue
            if dr in [EAST, WEST] and args.no_door_ew and args.no_open_ew:
                continue

            nbr = neighbor(cell, dr)
            if layout[cell][dr] == CLOSED:
                if nbr not in layout or not args.no_loops:
                    options.append((cell, dr))
    
    cell, dr = random.choice(options)
    nbr = neighbor(cell, dr)
    opp = opposite(dr)

    if dr in [NORTH, SOUTH] and args.no_door_ns and args.no_open_ns:
        raise RuntimeError('bad option')
    elif dr in [NORTH, SOUTH] and args.no_door_ns:
        connection = OPEN
    elif dr in [NORTH, SOUTH] and args.no_open_ns:
        connection = DOOR
    elif dr in [EAST, WEST] and args.no_door_ew and args.no_open_ew:
        raise RuntimeError('bad option')
    elif dr in [EAST, WEST] and args.no_door_ew:
        connection = OPEN
    elif dr in [EAST, WEST] and args.no_open_ew:
        connection = DOOR
    else:
        if random.randint(0, 99) < args.door_pct:
            connection = DOOR
        else:
            connection = OPEN

    layout[cell][dr] = connection

    if nbr not in layout:
        layout[nbr] = {}
        layout[nbr][NORTH] = CLOSED
        layout[nbr][SOUTH] = CLOSED
        layout[nbr][EAST] = CLOSED
        layout[nbr][WEST] = CLOSED

    layout[nbr][opp] = connection

cells = list(layout.keys())
print(layout)
x_lo = min(cells)[0]
x_hi = max(cells)[0]
y_lo = min(cells, key=lambda x: x[1])[1]
y_hi = max(cells, key=lambda x: x[1])[1]
print(x_lo, x_hi)
print(y_lo, y_hi)
width, height = abs(x_lo - x_hi)+1, abs(y_hi - y_lo)+1
x_adj, y_adj = abs(x_lo * 256 - 0), abs((y_lo * dims[0] * dims[1]) - 0)
print(width, height)
print(x_adj, y_adj)
#sys.exit()
layout_img = Image.new('RGB',(width*256, height*(dims[0]*dims[1])))
#img.save('test.png')

met_prob, zel_prob, mm_prob = args.met, args.zel, args.mm
layout_segments = {}
for key in layout:
    print(layout[key])
    cell = layout[key]
    label = [0] * 4
    if cell['north'] in ['open','door']:
        label[0] = 1
    if cell['south'] in ['open','door']:
        label[1] = 1
    if cell['west'] in ['open','door']:
        label[2] = 1
    if cell['east'] in ['open','door']:
        label[3] = 1
    print(label)
    z = sample_z()
    if 'blend' in GAME:
        game_label = [0,0]
        if random.random() > 0.5:
            game_label[0] = 1
        if random.random() > 0.5:
            game_label[1] = 1
        print('game: ', game_label)
        label = game_label + label
    label = get_label_tensor(label)
    if not args.multi:
        segment = get_segment_from_zc(model,z,label)
        img = get_image_from_segment(segment)
    else:
        r = random.random()
        #this_game = 'met' if r < met_prob else 'zelda'
        this_game = random.choices(['met','zelda','mm'], [met_prob, zel_prob, mm_prob])[0]
        segment = get_segment_from_zc(models[this_game],z,label,this_game)
        if this_game == 'zelda':
            segment = pad_zelda(segment)
        img = get_image_from_segment(segment,this_game)
    x, y = key
    layout_img.paste(img, ((x*256)+x_adj,(y*dims[0]*dims[1])+y_adj))
    print(x, y, '\t', (x*256)+x_adj, (y*dims[0]*dims[1])+y_adj)
    print('\n'.join(segment),'\n')
    layout_segments[key] = segment
if args.multi:
    layout_img.save('layout_multi_met' + str(met_prob) + '_zel' + str(zel_prob) + '_mm' + str(mm_prob) + '_' + str(latent_dim) + '.png')
else:
    layout_img.save('layout_' + GAME + '_' + str(latent_dim) + '.png')

if args.json:
    out_list = []
    for k, v in layout.items():
        out_dict = dict(v)
        out_dict['cell'] = k
        out_list.append(out_dict)
    out = json.dumps(out_list, sort_keys=True, indent=4)
    print(out)

else:
    for y in range(y_lo, y_hi + 1):
        line = ''
        for x in range(x_lo, x_hi + 1):
            cell = (x, y)
            if cell not in layout:
                line += '   '
            else:
                line += '┌'
                if (layout[cell][NORTH] == CLOSED):
                    line += '─'
                elif (layout[cell][NORTH] == DOOR):
                    line += '.'
                else:
                    line += ' '
                line += '┐'
        print(line)

        line = ''
        for x in range(x_lo, x_hi + 1):
            cell = (x, y)
            if cell not in layout:
                line += '   '
            else:
                if (layout[cell][WEST] == CLOSED):
                    line += '│'
                elif (layout[cell][WEST] == DOOR):
                    line += '.'
                else:
                    line += ' '
                line += ' '
                if (layout[cell][EAST] == CLOSED):
                    line += '│'
                elif (layout[cell][EAST] == DOOR):
                    line += '.'
                else:
                    line += ' '
        print(line)

        line = ''
        for x in range(x_lo, x_hi + 1):
            cell = (x, y)
            if cell not in layout:
                line += '   '
            else:
                line += '└'
                if (layout[cell][SOUTH] == CLOSED):
                    line += '─'
                elif (layout[cell][SOUTH] == DOOR):
                    line += '.'
                else:
                    line += ' '
                line += '┘'
        print(line)
