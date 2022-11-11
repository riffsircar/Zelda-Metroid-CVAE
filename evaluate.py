import argparse, random, torch, os, math, json, sys, re, pickle
import torch.optim as optim
import torch.utils.data
import numpy as np
from model_lin_cond import get_cond_model, load_cond_model
from sklearn.ensemble import RandomForestClassifier
from util import *
from tile_images import *
import warnings
from cvae_eval_metrics import *
import dcor

warnings.filterwarnings("ignore")

device = torch.device('cpu')
latent_dim = 32 # size of latent vector
GAME = 'lode'

zelda_folder = 'zelda_rooms_new/'
zelda_15_folder = 'zelda_padded/'
met_folder = 'met_chunks_all/'
lode_folder = 'lode_chunks/'
mm_folder = 'mm_chunks_all/'

folders = {'zelda':zelda_folder, 'met':met_folder, 'lode':lode_folder, 'mm':mm_folder}
out_folders = {'zelda':'out_zelda/','met':'out_met/','lode':'out_lode/','mm':'out_mm/'}
all_images = {'met':met_images, 'lode':lode_images, 'zelda':zelda_images, 'mm':mm_images, 'blend_met_mm':bmm_images, 'blend_zelda_lode':bzl_images}
images = all_images[GAME]
folder = folders[GAME] if 'blend' not in GAME else None
label_size = 4 if 'blend' not in GAME else 6
if 'blend' not in GAME:
	label_size = 4
elif GAME == 'blend_zelda_met_mm':
	label_size = 7
else:
	label_size = 6
num_labels = int(math.pow(2,label_size))

char2int, int2chars = char2ints[GAME], {}
for g in ['met','mm','zelda','lode','blend_zelda_mm','blend_zelda_met','blend_zelda_lode','blend_zelda_met_mm']:
	c2i = char2ints[g]
	i2c = {ch: ii for ii, ch in c2i.items()}
	int2chars[g] = i2c
int2char = int2chars[GAME]
num_tiles = len(char2int)

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

def classify(segment):
	out = []
	for s in segment:
		l = list(s)
		l_map = [char2int[x] for x in l]
		out.append(l_map)
	out = np.asarray(out)
	out_onehot = np.eye(num_tiles, dtype='uint8')[out]
	out_onehot = np.rollaxis(out_onehot, 2, 0)
	out_onehot = out_onehot[None, :, :]
	out = torch.DoubleTensor(out_onehot)
	out = out.to(device)
	out_lin = out.reshape(out.size(0),-1)
	out_lin = out_lin.to(torch.device('cpu')).numpy()
	pred = classifier.predict(out_lin)
	probs = classifier.predict_proba(out_lin)[0]
	return pred[0], probs[0]

# load CVAE
input_dim = 240 if 'met' in GAME or 'mm' in GAME else 176
model_name = 'models/cvae_lean_' + GAME + '_ld_' + str(latent_dim) + '_final.pth'
model = load_cond_model(model_name,input_dim,num_tiles,latent_dim,label_size,device)
model.eval()
model.to(device)


input_levels, input_text, input_dirs = parse_folder(folder,GAME,True)
if GAME != 'zelda':
    unique_dirs = [np.array(get_label_dir(d)).astype('uint8').tolist() for d in list(set(input_dirs))]
else:
    doors = [np.array(get_door_label(seg)).astype('uint8').tolist() for seg in input_levels]
    door_nums = [int("".join(str(x) for x in door), 2) for door in doors]
    dirs = set(door_nums)
    unique_dirs = [[int(x) for x in bin(dir)[2:].zfill(4)] for dir in dirs]
print('Unique Dirs: ', len(unique_dirs), unique_dirs[0])

input_set, inputs = set(), []
for level in input_levels:
	level_str = [''.join(l) for l in level]
	inputs.append(level_str)
	level_str = ''.join(level_str)
	input_set.add(level_str)
print(len(inputs))


# Density-Symmetry
if GAME in ['zelda','lode','blend_zelda_lode']:
	max_den = 11*16
	max_sym = (8*11)+(8*10)
else:
	max_den = 15*16
	max_sym = (8*15)+(8*14)
print(max_den, max_sym)
input_ds = []
outfile = open(GAME + '_orig_den_sym.csv','w')
outfile.write('Density,Symmetry\n')
in_d, in_s = [], []
for inp in inputs:
	d = density(GAME,inp)
	s = symmetry(inp)
	d /= max_den
	s /= max_sym
	in_d.append(d)
	in_s.append(s)
	input_ds.append([d,s])
	outfile.write(str(d) + ',' + str(s) + '\n')
outfile.close()

print('IN-Den: ', np.mean(in_d), np.std(in_d))
print('IN-Sym: ', np.mean(in_s), np.std(in_s))

output_ds = []
out_d, out_s = [], []
outfile = open(GAME + '_outs_ld_' + str(latent_dim) + '_den_sym.csv','w')
outfile.write('Density,Symmetry\n')
for i in range(len(input_ds)):
	z = sample_z()
	ds, ss = [], []
	for j in range(num_labels):
		label = bin(j)[2:].zfill(4)
		label = [int(x) for x in label]
		if label not in unique_dirs:
			continue
		label_tensor = get_label_tensor(label)
		segment = get_segment_from_zc(model,z,label_tensor)
		d, s = density(GAME,segment), symmetry(segment)
		d /= max_den
		s /= max_sym
		ds.append(d)
		ss.append(s)
	out_d.append(np.mean(ds))
	out_s.append(np.mean(ss))
	outfile.write(str(np.mean(ds)) + ',' + str(np.mean(ss)) + '\n')
	output_ds.append([np.mean(ds),np.mean(ss)])
outfile.close()
print('Out-Den: ', np.mean(out_d), np.std(out_d))
print('Out-Sym: ', np.mean(out_s), np.std(out_s))
print(len(input_ds),len(output_ds))
print("ED: ", dcor.energy_distance(input_ds,output_ds))
print("ED T-Test: ", dcor.homogeneity.energy_test(input_ds,output_ds,num_resamples=100))

# load classifier
classifier = None
if GAME != 'zelda':
	n_estimators = 1000 # 500, 750, 1000
	clf_name = 'classifiers/classifier_' + GAME + '_' + str(n_estimators) + '.pickle'
	clf_file = open(clf_name,'rb')
	classifier = pickle.load(clf_file)

# input_levels, input_text, input_dirs = parse_folder(folder,GAME,True)
# unique_dirs = list(set(input_dirs))

# Directional Label Accuracy
exact, ones, total = 0, 0, 0
for j in range(10):
	if j % 100 == 0:
		print(j)
	z = sample_z()
	for i in range(num_labels):
		total += 1
		label = bin(i)[2:].zfill(4)
		label = [int(x) for x in label]
		label_tensor = get_label_tensor(label)
		segment = get_segment_from_zc(model,z,label_tensor)
		pred, probs = classify(segment)
		pred_bin = bin(pred)[2:].zfill(4)
		pred_bin = [int(x) for x in pred_bin]
		if pred == i:
			exact += 1
			ones += 1
		else:
			valid = True
			for l,p in zip(label,pred_bin):
				if l == 1 and p == 0:
					valid = False
					break
			if valid:
				ones += 1
			else:
				pass
				pl = bin(pred)[2:].zfill(4)
				pl = [int(x) for x in pl]
print(exact, '\t', ones, '\t', total)