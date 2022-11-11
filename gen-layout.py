import argparse, random, torch, os, math, json, sys
import torch.utils.data
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model_lin_cond import get_cond_model, load_cond_model
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
parser.add_argument('--game', type=str, help='Game (options: met, mm, lode, zelda, blend_met_mm, blend_zelda_lode, blend_zelda_met, blend_zelda_mm, blend_zelda_met_mm)', default='met')
parser.add_argument('--ld', type=int, help='Latent dimension size', default=8)
parser.add_argument('--multi', action='store_true', help='Use multiple models (default: false)')
parser.add_argument('--met', type=float, help='met prob', default=0.33)
parser.add_argument('--zel', type=float, help='zelda prob', default=0.33)
parser.add_argument('--mm', type=float, help='mm prob', default=0.33)
parser.add_argument('--lode', type=float, help='lode prob', default=0.33)
parser.add_argument('--nolines', action='store_true',help='No lines (default:false)')
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
all_images = {'met':met_images, 'lode':lode_images, 'zelda':zelda_images, 'mm':mm_images, 'blend_met_mm':bmm_images, 'blend_zelda_lode':bzl_images,'blend_zelda_met':bzmet_images,'blend_zelda_mm':bzmm_images,'blend_zelda_met_mm':bzmetmm_images}
images = all_images[GAME]
folder = folders[GAME] if 'blend' not in GAME else None
if GAME == 'blend_zelda_met_mm':
    label_size = 7
elif 'blend' in GAME:
    label_size = 6
else:
    label_size = 4
print(label_size)
met_prob, zel_prob, mm_prob, lode_prob = args.met, args.zel, args.mm, args.lode

g1, g2 = None, None
if GAME == 'blend_zelda_met':
    g1, g2 = met_prob, zel_prob
elif GAME == 'blend_zelda_mm':
    g1, g2 = mm_prob, zel_prob
elif GAME == 'blend_zelda_met_mm':
    g1, g2, g3 = met_prob, mm_prob, zel_prob
elif GAME == 'blend_zelda_lode':
    g1, g2 = zel_prob, lode_prob
elif GAME == 'blend_met_mm':
    g1, g2 = met_prob, mm_prob

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

#chars = sorted(list(set(text.strip('\n'))))
#int2char = dict(enumerate(chars))
#char2int = {ch: ii for ii, ch in int2char.items()}
char2int, int2chars = char2ints[GAME], {}
for g in ['met','mm','zelda','lode','blend_zelda_lode','blend_met_mm','blend_zelda_met','blend_zelda_mm','blend_zelda_met_mm']:
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
        game_model = load_cond_model(model_name,input_dim,num_tiles,latent_dim,4,device)
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

def draw_lines(img,x,y,x_adj,y_adj,label):
    line_width=5
    dir_label = None
    if len(label) > 4:
        if len(label) == 6:
            gl = 2
            dir_label = label[2:]
        else:
            gl = 3
            dir_label = label[3:]
        dirs = ''
        if dir_label[0] == 1:
            dirs += 'U'
        if dir_label[1] == 1:
            dirs += 'D'
        if dir_label[2] == 1:
            dirs += 'L'
        if dir_label[3] == 1:
            dirs += 'R'
    x_pos, y_pos, x_del, y_del = (x*256)+x_adj, (y*dims[0]*dims[1])+y_adj, dims[1]*16, dims[0]*16
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('arial.ttf',size=25)
    draw.line(((x_pos,y_pos),(x_pos+x_del,y_pos)),width=line_width)  # up
    draw.line(((x_pos,y_pos+y_del),(x_pos+x_del,y_pos+y_del)),width=line_width)  # down
    draw.line(((x_pos,y_pos),(x_pos, y_pos+y_del)),width=line_width)  # left
    draw.line(((x_pos+x_del,y_pos),(x_pos+x_del,y_pos+y_del)),width=line_width)  # right
    #if dir_label:
        #draw.text((x_pos+(x_del/2.5),y_pos+(y_del/2.5)),str(label[:gl]) + '\n' + dirs,font=font,fill=(255,255,255,240),align='center',stroke_width=1)
    #    draw.text((x_pos+(x_del/2.5),y_pos+(y_del/2.5)),str(label[:gl]),font=font,fill=(255,255,255,240),align='center',stroke_width=1)
    #else:
    #    draw.text((x_pos+(x_del/5),y_pos+(y_del/2)),str(label),font=font,fill=(255,255,255,240),align='center',stroke_width=1)
    # doors/openings
    """
    if label[0] == 1:  # up
        draw.line((((x_pos+(x_del/3),y_pos),(x_pos+(x_del/1.5),y_pos))),width=line_width,fill=(255,0,0))
    if label[1] == 1:
        draw.line((((x_pos+(x_del/3),y_pos+y_del),(x_pos+(x_del/1.5),y_pos+y_del))),width=line_width,fill=(255,0,0))
    if label[2] == 1:
        draw.line((((x_pos,y_pos+(y_del/3)),(x_pos,y_pos+(y_del/1.5)))),width=line_width,fill=(255,0,0))
    if label[3] == 1:
        draw.line((((x_pos+x_del,y_pos+(y_del/3)),(x_pos+x_del,y_pos+(y_del/1.5)))),width=line_width,fill=(255,0,0))
    """

#draw_dirs(layout_img, xys, labels, imgs)
def draw_dirs(img, locs, labels):
    line_width=5
    draw = ImageDraw.Draw(img)
    x_del, y_del = dims[1]*16, dims[0]*16
    for (x_pos,y_pos),label in zip(locs,labels):
        dir_label = None
        if len(label) > 4:
            if len(label) == 6:
                gl = 2
                dir_label = label[2:]
            else:
                gl = 3
                dir_label = label[3:]
            dirs = ''
            if dir_label[0] == 1:
                dirs += 'U'
            if dir_label[1] == 1:
                dirs += 'D'
            if dir_label[2] == 1:
                dirs += 'L'
            if dir_label[3] == 1:
                dirs += 'R'
               
        if dir_label:
            label = dir_label
        if label[0] == 1:  # up
            draw.line((((x_pos+(x_del/3),y_pos),(x_pos+(x_del/1.5),y_pos))),width=line_width,fill=(255,0,0))
        if label[1] == 1:
            draw.line((((x_pos+(x_del/3),y_pos+y_del),(x_pos+(x_del/1.5),y_pos+y_del))),width=line_width,fill=(255,0,0))
        if label[2] == 1:
            draw.line((((x_pos,y_pos+(y_del/3)),(x_pos,y_pos+(y_del/1.5)))),width=line_width,fill=(255,0,0))
        if label[3] == 1:
            draw.line((((x_pos+x_del,y_pos+(y_del/3)),(x_pos+x_del,y_pos+(y_del/1.5)))),width=line_width,fill=(255,0,0))

if args.no_door_ns and args.no_open_ns and args.no_door_ew and args.no_open_ew:
    raise RuntimeError('no way to connect')

"""
input_levels, input_text, input_dirs = parse_folder(folder,GAME,True)
inputs = []
for level in input_levels:
	level_str = [''.join(l) for l in level]
	inputs.append(level_str)
#idx = 3
#img = get_image_from_segment(inputs[idx])
#img.save(GAME + str(latent_dim) + '_' + str(idx) + '_' + input_dirs[idx] + '_test.png')
#img.save(GAME + str(latent_dim) + '_' + str(idx) + '_test.png')
#sys.exit()
"""
rooms = random.randint(args.min_rooms, args.max_rooms)
if GAME == 'mmsad':
    segments = random.randint(10,15)
    prev = None
    if random.random() < 0.8:
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
    imgs, labels, xys = [], [], []
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
        label_tensor = get_label_tensor(label)
        segment = get_segment_from_zc(model,z,label_tensor)
        img = get_image_from_segment(segment)
        draw = ImageDraw.Draw(img)
        x, y = key
        
        imgs.append(img)
        labels.append(label)
        x_pos, y_pos, x_del, y_del = (x*256)+x_adj, (y*dims[0]*dims[1])+y_adj, dims[1]*16, dims[0]*16
        
        xys.append((x_pos,y_pos))
        #draw.rectangle((x_pos,y_pos,x_pos+(dims[1]*16),y_pos+(dims[0]*16)),outline=(255,255,255))
        #draw.rectangle(((x*256)+x_adj, (y*dims[0]*dims[1])+y_adj))
        #layout_img.paste(img, ((x*256)+x_adj,(y*dims[0]*dims[1])+y_adj))
        layout_img.paste(img, (x_pos,y_pos))
        #draw.line(((x_pos,y_pos),(x_pos+x_del,y_pos)),width=3)  # up
        #draw.line(((x_pos,y_pos+y_del),(x_pos+x_del,y_pos+y_del)),width=3)  # down
        #draw.line(((x_pos,y_pos),(x_pos, y_pos+y_del)),width=3)  # left
        #draw.line(((x_pos+x_del,y_pos),(x_pos+x_del,y_pos+y_del)),width=3)  # right
        if not args.nolines:
            draw_lines(layout_img,x,y, x_adj, y_adj,label)
        print(x, y, '\t', (x*256)+x_adj, (y*dims[0]*dims[1])+y_adj)
        print('\n'.join(segment),'\n')
    if not args.nolines:
        draw_dirs(layout_img, xys, labels)
    layout_img.save('layout_' + GAME + '_' + str(latent_dim) + '.png')
    sys.exit()
elif GAME == 'lode':
    rows, cols = random.randint(2,2), random.randint(2,2)
    layout_img = Image.new('RGB',(cols*256, rows*(dims[0]*dims[1])))
    imgs, labels, xys = [], [], []
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
            label_tensor = get_label_tensor(label)
            segment = get_segment_from_zc(model,z,label_tensor)
            img = get_image_from_segment(segment)
            layout_img.paste(img, ((x*256),(y*dims[0]*dims[1])))
            imgs.append(img)
            labels.append(label)
            x_pos, y_pos, x_del, y_del = (x*256), (y*dims[0]*dims[1]), dims[1]*16, dims[0]*16
        
            xys.append((x_pos,y_pos))
            layout_img.paste(img, (x_pos,y_pos))    
            #draw.line(((x_pos,y_pos),(x_pos+x_del,y_pos)),width=3)  # up
            #draw.line(((x_pos,y_pos+y_del),(x_pos+x_del,y_pos+y_del)),width=3)  # down
            #draw.line(((x_pos,y_pos),(x_pos, y_pos+y_del)),width=3)  # left
            #draw.line(((x_pos+x_del,y_pos),(x_pos+x_del,y_pos+y_del)),width=3)  # right
            if not args.nolines:
                draw_lines(layout_img,x,y, 0, 0,label)
            
    if not args.nolines:
        draw_dirs(layout_img, xys, labels)
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
#draw = ImageDraw.Draw(layout_img)
#img.save('test.png')

layout_segments = {}
imgs, labels, xys = [], [], []
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
        if GAME == 'blend_zelda_met_mm':
            game_label = [0,0,0]
            if random.random() <= g1:
                game_label[0] = 1
            if random.random() <= g2:
                game_label[1] = 1
            if random.random() <= g3:
                game_label[2] = 1
        else:
            game_label = [0,0]
            if random.random() <= g1:
                game_label[0] = 1
            if random.random() <= g2:
                game_label[1] = 1
            print('game: ', game_label)
            
    if not args.multi:
        if 'blend' in GAME:
            label = game_label + label
        label_tensor = get_label_tensor(label)
        segment = get_segment_from_zc(model,z,label_tensor)
        img = get_image_from_segment(segment)
    else:
        label_tensor = get_label_tensor(label)
        r = random.random()
        #this_game = 'met' if r < met_prob else 'zelda'
        this_game = random.choices(['met','zelda','mm'], [met_prob, zel_prob, mm_prob])[0]
        print(this_game)
        segment = get_segment_from_zc(models[this_game],z,label_tensor,this_game)
        if this_game == 'zelda':
            segment = pad_zelda(segment)
        img = get_image_from_segment(segment,this_game)
    x, y = key
    imgs.append(img)
    labels.append(label)
    x_pos, y_pos, x_del, y_del = (x*256)+x_adj, (y*dims[0]*dims[1])+y_adj, dims[1]*16, dims[0]*16
    xys.append((x_pos,y_pos))
    layout_img.paste(img, (x_pos,y_pos))
    if not args.nolines:
        draw_lines(layout_img,x,y, x_adj, y_adj,label)
    print(x, y, '\t', (x*256)+x_adj, (y*dims[0]*dims[1])+y_adj)
    print('\n'.join(segment),'\n')
    layout_segments[key] = segment
if not args.nolines:
    draw_dirs(layout_img, xys, labels)
if args.multi:
    layout_img.save('layout_multi_met' + str(met_prob) + '_zel' + str(zel_prob) + '_mm' + str(mm_prob) + '_' + str(latent_dim) + '.png')
else:
    if 'blend' not in GAME:
        layout_img.save('layout_' + GAME + '_' + str(latent_dim) + '.png')
    elif GAME == 'blend_zelda_met_mm':
        layout_img.save('layout_' + GAME + '_' + str(g1) + '_' + str(g2) + '_' + str(g3) + '_' + str(latent_dim) + '.png')
    else:
        layout_img.save('layout_' + GAME + '_' + str(g1) + '_' + str(g2) + '_' + str(latent_dim) + '.png')
    """
    elif GAME == 'blend_zelda_met':
        layout_img.save('layout_' + GAME + '_' + str(met_prob) + '_zel' + str(zel_prob) + '_' + str(latent_dim) + '.png')
    elif GAME == 'blend_met_mm':
        layout_img.save('layout_' + GAME + '_' + str(met_prob) + '_mm' + str(mm_prob) + '_' + str(latent_dim) + '.png')
    elif GAME == 'blend_zelda_mm':
        layout_img.save('layout_' + GAME + '_zel' + str(zel_prob) + '_mm' + str(mm_prob) + '_' + str(latent_dim) + '.png')
    elif GAME == 'blend_zelda_met_mm':
        layout_img.save('layout_' + GAME + '_met' + str(met_prob) + '_zel' + str(zel_prob) + '_mm' + str(mm_prob) + '_' + str(latent_dim) + '.png')
    elif GAME == 'blend_zelda_lode':
        layout_img.save('layout_' + GAME + '_zel' + str(zel_prob) + '_lode' + str(lode_prob) + '_' + str(latent_dim) + '.png')
    """
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
