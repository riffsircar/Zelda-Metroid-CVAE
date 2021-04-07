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

device = torch.device('cpu')
latent_dim = 8 # size of latent vector
GAME = 'lode'  # smb, ki, mm, smb_pats, all

label_sizes = {'dg':4,'zelda':4,'met':4,'lode':4,'blend_dz':16, 'blend_dz_2': 2, 'blend_dz_common':2,'blend_dz_common_full':16, 'blend_dz_common_doors':6}
label_size = label_sizes[GAME]
num_labels = int(math.pow(2,label_size))

dims = (15,16) if GAME == 'met' else (11,16)

zelda_folder = 'zelda_rooms/'
dg_folder = 'dg_chunks_edited/'
met_folder = 'met_chunks_all/'
lode_folder = 'lode_chunks/'

dg_mean = 0.01 #0.007690180346566041 #-0.013997895605728334
dg_std = 0.18 #0.1813693650824676 #0.17229445236817564
zelda_mean = -0.02 #-0.015203464839032359 #0.01327264206087219
zelda_std = 0.17 #0.17129609054447595 #0.1696900470971206 

folders = {'dg':dg_folder,'zelda':zelda_folder,'blend_dz':None, 'blend_dz_2':None, 'blend_dz_common':None,
		   'blend_dz_common_full':None, 'met':met_folder, 'lode':lode_folder}
out_folders = {'zelda':'out_zelda/','met':'out_met/','lode':'out_lode/'}
all_images = {'met':met_images, 'lode':lode_images, 'zelda':zelda_images}
images = all_images[GAME]
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
	
def parse_folder(folder):
	levels, text = [], ''
	files = os.listdir(folder)
	files[:] = (value for value in files if value != '.')
	files = natural_sort(files)
	for file in files:
		if file.startswith('.'):
			continue
		with open(os.path.join(folder,file),'r') as infile:
			level = []
			for line in infile:
				if GAME.startswith('blend'):
					line = line.replace('F','-')
					line = line.replace('W','X')
				text += line
				level.append(list(line.rstrip()))
			levels.append(level)
	return levels, text

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

chars = sorted(list(set(text.strip('\n'))))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
print(char2int)
num_tiles = len(char2int)
#print(num_tiles)

input_levels = set()
for level in levels:
	level_string = ''
	l = [''.join(line) for line in level]
	l_string = '\n'.join(l)
	input_levels.add(l_string)
print(len(input_levels))


level_rows, level_cols = set(), set()
if GAME == 'met':
	for level in levels:
		level_t = [[level[j][i] for j in range(len(level))] for i in range(len(level[0]))]
		for l in level:
			l_str = ''.join(l)
			#if l_str not in level_rows:
			level_rows.add(l_str)
		for lt in level_t:
			lt_str = ''.join(lt)
			#if lt_str not in level_cols:
			level_cols.add(lt_str)
	print('Rows: ', len(level_rows))
	print('Cols: ', len(level_cols))

def write_segment_to_file(segment,name):
    outfile = open(out_folders[GAME] + '/' + name + '.txt','w')
    for row in segment:
        outfile.write(row + '\n')
    outfile.close()

def get_image_from_segment(segment,name):
    img = Image.new('RGB',(dims[1]*16, dims[0]*16))
    for row, seq in enumerate(segment):
        for col, tile in enumerate(seq):
            img.paste(images[tile],(col*16,row*16))
    img.save(out_folders[GAME] + '/' + name + '.png')


def get_segment_from_file(folder,f):
    chunk = open(folder + f, 'r').read().splitlines()
    chunk = [line.replace('\r\n','') for line in chunk]
    return chunk
    out = []
    for line in chunk:
        line_list = list(line)
        #line_list_map = [char2int[x] for x in line_list]
        out.append(line_list)
    return out

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

def get_label_tensor(label):
	return torch.DoubleTensor(label).reshape(1,-1).to(device)

get_label_func = {'dg':get_label_dg, 'zelda':get_label_zelda, 'blend_dz':get_label_blend, 'blend_dz_2': None,
				  'blend_dz_common_full':get_label_blend, 'blend_dz_common_doors':get_label_blend,'met':get_label_met,'lode':None}
if not GAME.startswith('blend'):
	get_label = get_label_func[GAME]

input_dim = 240 if GAME == 'met' else 176
model_name = 'models/cvae_lean_' + GAME + '_ld_' + str(latent_dim) + '_final.pth'
model = load_cond_model(model_name,input_dim,num_tiles,latent_dim,label_size,device)
model.eval()

content_label = [1,1,1,0,1,0,0,0,0,1]
zelda_label = [1,0]
dg_label = [0,1]
blend_label = [1,1]
door_label = [0,0,0,1]

content = [0,0,0,1,0]
totals = []


copies, noncopies = 0, 0
for j in range(1):
	if j % 100 == 0:
		print(j)
	z = torch.DoubleTensor(1,latent_dim).normal_(0,1).to(device)
	cont = bin(j)[2:].zfill(5)
	cont = [int(x) for x in cont]
	cont = [1,0,0,0,0]
	total = 0
	for i in range(16):
		doors = bin(i)[2:].zfill(4)
		doors = [int(x) for x in doors]
		label = doors #+ cont
		#print(label)
		label_tensor = get_label_tensor(label)
		segment = get_segment_from_zc(z,label_tensor)
		seg_string = '\n'.join(segment)
		#print(seg_string)
		#print(seg_string in input_levels,'\n')
		if seg_string in input_levels:
			copies += 1
			print(label, ' copy')
		else:
			noncopies += 1
			print(seg_string)
			print(label, ' noncopy')
		
		get_image_from_segment(segment,str(label))
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

print(copies, '\t', noncopies)
#print(totals)
#print(np.mean(totals))
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

"""
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
"""