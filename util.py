import re

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

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

char2ints = {
    'met': {'#': 0, '+': 1, '-': 2, 'B': 3, 'D': 4, 'E': 5, '^': 6},
    'mm': {'#': 0, '*': 1, '+': 2, '-': 3, 'B': 4, 'C': 5, 'H': 6, 'L': 7, 'M': 8, 'U': 9, 'W': 10, 'l': 11, 't': 12, 'w': 13, '|': 14},
    'lode': {'-': 0, 'B': 1, 'E': 2, 'G': 3, 'L': 4, 'M': 5, 'R': 6, 'b': 7},
    'zelda': {'B': 0, 'D': 1, 'F': 2, 'I': 3, 'M': 4, 'O': 5, 'P': 6, 'S': 7, 'W': 8},
    'blend_met_mm': {'#': 0, '*': 1, '+': 2, '-': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'H': 8, 'L': 9, 'M': 10, 'U': 11, 'W': 12, '^': 13, 'l': 14, 't': 15, 'w': 16, '|': 17},
    'blend_zelda_lode': {'#': 0, '-': 1, 'B': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'I': 7, 'L': 8, 'M': 9, 'N': 10, 'O': 11, 'P': 12, 'R': 13, 'S': 14, 'W': 15, 'b': 16},
    'blend_zelda_met': {'#': 0, '+': 1, '-': 2, 'B': 3, 'D': 4, 'E': 5, 'F': 6, 'I': 7, 'K': 8, 'M': 9, 'O': 10, 'P': 11, 'S': 12, 'W': 13, '^': 14},
    'blend_zelda_mm': {'#': 0, '*': 1, '+': 2, '-': 3, 'B': 4, 'C': 5, 'D': 6, 'F': 7, 'H': 8, 'I': 9, 'K': 10, 'L': 11, 'M': 12, 'O': 13, 'P': 14, 'S': 15, 'U': 16, 'W': 17, 'l': 18, 'm': 19, 't': 20, 'w': 21, '|': 22},
    'blend_zelda_met_mm':{'#': 0, '*': 1, '+': 2, '-': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'H': 9, 'I': 10, 'K': 11, 'L': 12, 'M': 13, 'O': 14, 'P': 15, 'S': 16, 'U': 17, 'W': 18, '^': 19, 'l': 20, 'm': 21, 't': 22, 'w': 23, '|': 24}
}

densities = {
    'met': ['#', 'B', 'D'],
    'mm': ['#', 'B', 'C', 'D', 'H', '|', 'M'],
    'lode': ['B', 'b', 'L', 'R'],
    'zelda': ['B', 'D', 'W', 'I', 'M', 'S'],
    'blend_met_mm': ['#', 'B', 'D', 'C', 'D', 'H', '|', 'M'],
    'blend_zelda_lode': ['B', 'D', 'I', 'M', 'S', 'W', '#', 'b', 'L', 'R'],
    'blend_zelda_met': ['K', 'I', 'M', 'S', 'W', '#', 'B', 'D'],
    'blend_zelda_mm': ['K', 'I', 'M', 'S', 'W', '#', 'B', 'C', 'D', 'H', 'm', '|'],
    'blend_zelda_met_mm':['K', 'I', 'M', 'S', 'W', '#', 'B', 'C', 'D', 'H', 'm', '|']
}


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
    #print(len(padded))
    #print('\n'.join(padded))
    #print('\n','\n'.join(segment))
    return padded