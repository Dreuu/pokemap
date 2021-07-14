import nlzss.lzss3
from PIL import Image
import struct
import sys
import argparse
import os
from multiprocessing import Pool
import numpy as np
import time


def debug(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)


def main():
    global THREAD_COUNT
    global DEBUG_MODE
    global GAME
    global frOffsets
    global lgOffsets
    global emOffsets
    global ruOffsets
    global saOffsets
    global gameIdOffset

    THREAD_COUNT = None
    DEBUG_MODE = False
    frOffsets = {"strings": "0x3eecfc", "banks": "0x5524C"}
    lgOffsets = {"strings": "0x3eecfc", "banks": "0x55238"}  # Wrong strings offset
    emOffsets = {"strings": "0x5A0B2C", "banks": "0x84AA4"}
    ruOffsets = {"strings": "0x5A0B2C", "banks": "0x53324"}  # Wrong strings offset
    saOffsets = {"strings": "0x5A0B2C", "banks": "0x53310"}  # Wrong strings offset
    gameIdOffset = "0xAC"
    chosenROM = None

    parser = argparse.ArgumentParser(
        description="do a wee bit o' data rippin from a rom"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print information while running to help debug",
        action="store_true",
        dest="verbose",
    )
    parser.add_argument(
        "rom_file", metavar="rom", type=str, help="a fire red or emerald rom"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Specify the output file name/extension",
        action=CheckExt({"png", "bmp", "jpeg", "tga"}),
        default=None,
    )
    parser.add_argument(
        "-g",
        "--game",
        help="Specify which game the ROM contains. Options are 'fr' and 'em'",
        default=None,
    )
    parser.add_argument(
        "-t", "--thread_count", help="Number of CPU threads to use", default=None
    )
    parser.add_argument(
        "-d",
        "--draw",
        help="Which maps to draw. Options are 'all' for all maps, and 'group' \
                            for only map groups. If given any other value, only the \
                            largest map group will be drawn (Default)",
        default="",
    )
    args = parser.parse_args()

    GAME = args.game

    if args.thread_count is not None:
        THREAD_COUNT = int(args.thread_count)
    else:
      print("Using option '-t THREAD_COUNT' is highly recommended to improve performance!")

    if args.verbose:
        DEBUG_MODE = True

    print()
    print("Reading ROM:", args.rom_file)
    bytes = load_rom(args.rom_file)

    # Set default GAME type based on hex header
    # BPRE, BPGE, BPEE...
    if GAME is None:
        game_id = read_uint(bytes, gameIdOffset)
        game_id = hex(game_id)
        if game_id == "0x45525042":
            GAME = "fr"
        if game_id == "0x45475042":
            GAME = "lg"
        elif game_id == "0x45455042":
            GAME = "em"
        elif game_id == "0x45565841":
            GAME = "ru"
        elif game_id == "0x45505841":
            GAME = "sa"

    # Set offset of map groups
    if GAME == "fr":
        chosenROM = frOffsets
    elif GAME == "lg":
        chosenROM = lgOffsets
    elif GAME == "em":
        chosenROM = emOffsets
    elif GAME == "ru":
        chosenROM = ruOffsets
    elif GAME == "sa":
        chosenROM = saOffsets

    # Set default output filename
    if args.outfile is None:
        args.outfile = f"{GAME}_map.png"

    # strings = load_strings(bytes, chosenROM['strings'])
    # debug('Found all these strings: {}'.format([x.encode('utf-8') for x in strings]))
    bank_pointer = hex(
        read_pointer(bytes, int(chosenROM["banks"], 16))
    )  # Automatically find pointer
    banks = load_maps(bytes, bank_pointer)

    group_list, single_list = get_seeds(banks, args.draw)
    group_count = len(group_list)
    for group_num, (map_bank, map_number) in enumerate(group_list, start=1):
      print(f"\nDrawing Group {group_num}/{group_count}")
      draw_maps((map_bank, map_number), banks, bytes, group_num, args, True)

    single_count = len(single_list)
    rjust = len(str(single_count))
    if THREAD_COUNT is None:
      pic_num = 1
      for idx, (map_bank, map_number) in enumerate(single_list, start=1):
        #print(f"Drawing Map {map_bank}.{map_number}")
        print(f"-{single_count-idx} maps left{' '*5}", end='\r')
        pic_num += draw_maps((map_bank, map_number), banks, bytes, str(pic_num).rjust(rjust,'0'), args, False)
    else:
        inputs = [((map_bank, map_number), banks, bytes, str(idx).rjust(rjust,'0'), args, False) 
                  for idx, (map_bank, map_number) in enumerate(single_list, start=1)]
        with Pool(THREAD_COUNT) as p:
          p.starmap(draw_maps, inputs)
        

def draw_maps(map_addr, banks, bytes, num, args, is_group):
  map_bank, map_number = map_addr
  offsets = calculate_map_offsets(banks, map_bank, map_number)

  map_ims, coords, width, height = get_map_parts(banks, bytes, offsets, is_group)

  if None in [map_ims, coords, width, height]:
      return 0

  map_im = Image.new(mode="RGBA", size=(width, height), color=(64, 64, 64, 255))

  for idx, im in enumerate(map_ims):
    if im is None:
        continue
    x, y = coords[idx]
    map_im.paste(im, (x, y))

  outfile = args.outfile.split(".")
  ext = outfile[-1]
  dir = outfile[0]
  outfile = outfile[:-1]
  outfile = "".join(outfile)
  if not is_group:
    outfile += f"_{num}-{map_bank}_{map_number}"
  if args.draw.upper() in ['A', 'G', 'ALL', 'GROUP', 'GROUPS']:
    if not os.path.exists(args.outfile.split(".")[0]):
      os.makedirs(args.outfile.split(".")[0])
    outfile = dir + "/" + outfile
    if is_group:
      outfile += "-group" + str(num)
  outfile += "." + ext
  if ext == "jpeg":  # No transparency
    map_im = map_im.convert("RGB")
  print("Saving Image:", outfile)
  try:
    map_im.save(outfile, outfile.split(".")[-1])
    return 1
  except:
    return 0


def get_map_parts(banks, bytes, offsets, is_group):
  try:
    min_x = min([x for ((m, b), (x, y)) in offsets])
    min_y = min([y for ((m, b), (x, y)) in offsets])
  except ValueError as e:
    return None, None, None, None

  max_x = max([x + banks[m][b]["width"] for ((m, b), (x, y)) in offsets])
  max_y = max([y + banks[m][b]["height"] for ((m, b), (x, y)) in offsets])

  width = (max_x - min_x) * 16
  height = (max_y - min_y) * 16
  if width > 16000 or height > 16000:
    return None, None, None, None
  x_orig = min_x * 16
  y_orig = min_y * 16

  map_ims = []
  coords = []
  if THREAD_COUNT is not None and len(offsets) > 1 and is_group:
    print(f'{len(offsets)} Maps')
    inputs = []
    for idx, ((m, b), (x, y)) in enumerate(offsets, start=1):
      inputs.append((bytes, banks[m][b]["map_data"], (m,b)))
      coords.append(((x - min_x) * 16, (y - min_y) * 16))
    with Pool(THREAD_COUNT) as p:
      map_ims = p.starmap(draw_map, inputs)
  else:
    current = 1
    offset_len = len(offsets)
    for idx, ((m, b), (x, y)) in enumerate(offsets, start=1):
      if is_group:
        print(f"Drawing Map {m}.{b}")
        print(f"-{offset_len-idx} maps left", end='\r')
      map_ims.append(draw_map(bytes, banks[m][b]["map_data"]))
      coords.append(((x - min_x) * 16, (y - min_y) * 16))
      current += 1
  for idx in range(len(map_ims)):
    if map_ims[idx] is not None:
        map_ims[idx] = Image.fromarray(map_ims[idx], "RGBA")
  return map_ims, coords, width, height


def load_rom(rom_path):
    return open(rom_path, "rb").read()


def load_strings(bytes, hex_offset):
    offset = int(hex_offset, 16)
    table = {
      0x00:' ',0x01:'À',0x02:'Á',0x03:'Â',0x04:'Ç',0x05:'È',0x06:'É',0x07:'Ê',
      0x08:'Ë',0x09:'Ì',0x0B:'Î',0x0C:'Ï',0x0D:'Ò',0x0E:'Ó',0x0F:'Ô',0x10:'Œ',
      0x11:'Ù',0x12:'Ú',0x13:'Û',0x14:'Ñ',0x15:'ß',0x16:'à',0x17:'á',0x19:'ç',
      0x1A:'è',0x1B:'é',0x1C:'ê',0x1D:'ë',0x1E:'ì',0x20:'î',0x21:'ï',0x22:'ò',
      0x23:'ó',0x24:'ô',0x25:'œ',0x26:'ù',0x27:'ú',0x28:'û',0x29:'ñ',0x2A:'º',
      0x2B:'ª',0x2D:'&',0x2E:'+',0x34:'[Lv]',0x35:'=',0x36:';',0x51:'¿',0x52:'¡',
      0x53:'[pk]',0x54:'[mn]',0x55:'[po]',0x56:'[ké]',0x57:'[bl]',0x58:'[oc]',
      0x59:'[k]',0x5A:'Í',0x5B:'%',0x5C:'(',0x5D:')',0x68:'â',0x6F:'í',0x79:'[U]',
      0x7A:'[D]',0x7B:'[L]',0x7C:'[R]',0x85:'<',0x86:'>',0xA1:'0',0xA2:'1',
      0xA3:'2',0xA4:'3',0xA5:'4',0xA6:'5',0xA7:'6',0xA8:'7',0xA9:'8',0xAA:'9',
      0xAB:'!',0xAC:'?',0xAD:'.',0xAE:'-',0xAF:'·',0xB0:'...',0xB1:'«',0xB2:'»',
      0xB3:'\'',0xB4:'\'',0xB5:'|m|',0xB6:'|f|',0xB7:'$',0xB8:',',0xB9:'*',
      0xBA:'/',0xBB:'A',0xBC:'B',0xBD:'C',0xBE:'D',0xBF:'E',0xC0:'F',0xC1:'G',
      0xC2:'H',0xC3:'I',0xC4:'J',0xC5:'K',0xC6:'L',0xC7:'M',0xC8:'N',0xC9:'O',
      0xCA:'P',0xCB:'Q',0xCC:'R',0xCD:'S',0xCE:'T',0xCF:'U',0xD0:'V',0xD1:'W',
      0xD2:'X',0xD3:'Y',0xD4:'Z',0xD5:'a',0xD6:'b',0xD7:'c',0xD8:'d',0xD9:'e',
      0xDA:'f',0xDB:'g',0xDC:'h',0xDD:'i',0xDE:'j',0xDF:'k',0xE0:'l',0xE1:'m',
      0xE2:'n',0xE3:'o',0xE4:'p',0xE5:'q',0xE6:'r',0xE7:'s',0xE8:'t',0xE9:'u',
      0xEA:'v',0xEB:'w',0xEC:'x',0xED:'y',0xEE:'z',0xEF:'|>|',0xF0:':',0xF1:'Ä',
      0xF2:'Ö',0xF3:'Ü',0xF4:'ä',0xF5:'ö',0xF6:'ü',0xF7:'|A|',0xF8:'|V|',
      0xF9:'|<|',0xFA:'|nb|',0xFB:'|nb2|',0xFC:'|FC|',0xFD:'|FD|',0xFE:'|br|',
    }
    strings = []
    string = ""
    while True:
        char = struct.unpack("<B", bytes[offset : (offset + 1)])[0]
        offset = offset + 1
        if char == 0xFF:
            strings.append(string)
            string = ""
            continue
        elif char not in table:
            break
        string += table[char]
    return strings


def load_maps(bytes, hex_offset):
    offset = int(hex_offset, 16)

    bank_pointers = []
    while is_pointer(bytes, offset):
        bank_pointer = read_pointer(bytes, offset)
        bank_pointers.append(bank_pointer)
        offset = offset + 4

    debug("Found these bank pointers: {}".format(bank_pointers))

    banks = []
    increment = 0
    for i, bank_pointer in enumerate(bank_pointers):
        if bank_pointer == 0:
            continue
        increment += 1
        offset = bank_pointer
        next_pointer = bank_pointers[i + 1] if (i + 1) < len(bank_pointers) else 0
        if increment == 1000:  # 42
            break

        maps = []
        while is_pointer(bytes, offset):
            map_data = read_pointer(bytes, offset)
            if map_data < 0:
                continue
            cs = read_connections(bytes, map_data)

            map_pointer = read_pointer(bytes, map_data)
            width = read_int(bytes, map_pointer)
            height = read_int(bytes, map_pointer + 4)

            maps.append(
                {
                    "map_data": map_data,
                    "connections": cs,
                    "width": width,
                    "height": height,
                }
            )

            offset = offset + 4
            if offset == next_pointer:
                break

        debug("Found {} map pointers: {}".format(len(maps), maps))
        banks.append(maps)

    return banks


def get_seeds(map_data, draw):
    """Find groups of maps and return a single bank/number combo for each"""
    class Graph:
        def __init__(self):
            self.adj = {}

        # Find all connected maps
        def DFSUtil(self, temp, i, visited):
            visited[i] = True
            temp.append(i)
            for j in self.adj[i]:
                if visited[j] == False:
                    temp = self.DFSUtil(temp, j, visited)
            return temp

        def addMap(self, v, w=None):
            if v not in self.adj:
                self.adj[v] = []
            if w is not None:
                self.adj[v].append(w)
                if w not in self.adj:
                    self.adj[w] = []
                self.adj[w].append(v)

        # Find all groups of connected maps
        def connectedComponents(self):
            visited = {}
            groups = []
            for i in self.adj:
                visited[i] = False
            for i in self.adj:
                v = visited[i]
                if not v:
                    temp = []
                    groups.append(self.DFSUtil(temp, i, visited))
            return groups

    connections = [[y["connections"] for y in x] for x in map_data]
    single_maps = []
    g = Graph()
    for map_bank in range(len(connections)):
        for map_number in range(len(connections[map_bank])):
            if len(connections[map_bank][map_number]) > 0:
                for connection in connections[map_bank][map_number]:
                    to_bank = connection["map_bank"]
                    to_number = connection["map_number"]
                    g.addMap((map_bank, map_number), (to_bank, to_number))
                    g.addMap((to_bank, to_number))  # Put a blank entry in adj
            else:
                width = map_data[map_bank][map_number]["width"]
                height = map_data[map_bank][map_number]["height"]
                if width <= 16000 and height <= 16000:
                    single_maps.append((map_bank, map_number, width * height))

    single_maps.sort(reverse=True, key=lambda x: x[2])
    single_maps = [(x[0], x[1]) for x in single_maps]
    groups = g.connectedComponents()
    groups.sort(reverse=True, key=lambda x: len(x))
    groups = [x[0] for x in groups]

    if draw.upper() in ["ALL","A"]:
        return (groups, single_maps)
    elif draw.upper() in ["G","GROUP","GROUPS"]:
        return (groups, [])
    else:
        return ([groups[0]], [])


def read_connections(bytes, map_data):
    cs = []
    if not is_pointer(bytes, map_data + 12):
        return cs
    connections = read_pointer(bytes, map_data + 12)
    n_connections = read_int(bytes, connections)
    offset = read_pointer(bytes, connections + 4)
    if n_connections > 10 or offset < 0:
        return cs
    debug("Reading connections at {:#x}".format(offset))
    for i in range(n_connections):
        cs.append(
            {
                "direction": read_int(bytes, offset),
                "offset": read_uint(bytes, offset + 4),
                "map_bank": struct.unpack("<B", bytes[(offset + 8) : (offset + 9)])[0],
                "map_number": struct.unpack("<B", bytes[(offset + 9) : (offset + 10)])[
                    0
                ],
            }
        )
        offset += 12
    return cs


def read_map(bytes, header_pointer):
    map_pointer = read_pointer(bytes, header_pointer)
    width = read_int(bytes, map_pointer)
    height = read_int(bytes, map_pointer + 4)
    label = struct.unpack("<B", bytes[(header_pointer + 20) : (header_pointer + 21)])[0]
    border = read_pointer(bytes, map_pointer + 8)
    tiles_pointer = read_pointer(bytes, map_pointer + 12)
    tileset_pointer = read_pointer(bytes, map_pointer + 16)
    local_pointer = read_pointer(bytes, map_pointer + 20)

    debug("Map at {}".format(map_pointer))
    debug("Width/height: {}/{}".format(width, height))
    debug("Border pointer: {0:#x}, tiles pointer: {1:#x}".format(border, tiles_pointer))

    tile_sprites = {}

    offset = tiles_pointer
    i = 0
    for y in range(height):
        for x in range(width):
            try:
                tile_data = struct.unpack(
                    "<H", bytes[(offset + i * 2) : (offset + (i + 1) * 2)]
                )[0]
            except:
                return [None] * 6
            attribute = tile_data >> 10
            tile = tile_data & 0x3FF
            debug(
                "tile at ({}, {}): {:#x}, attribute: {:#x}".format(
                    x, y, tile, attribute
                )
            )
            tile_sprites[(x, y)] = tile

            i = i + 1

    return (width, height, label, tile_sprites, tileset_pointer, local_pointer)


def read_tileset(bytes, tileset_pointer):
    attribs = struct.unpack("<2B", bytes[tileset_pointer : (tileset_pointer + 2)])
    debug("Tileset compressed: {}, primary: {}".format(attribs[0], attribs[1]))
    primary = attribs[1]
    tileset_image_pointer = read_pointer(bytes, tileset_pointer + 4)
    try:
        image = nlzss.lzss3.decompress_bytes(bytes[tileset_image_pointer:])
    except:
        return (None, None, None)

    tiles = []
    for i in range(0, len(image), 32):
        tile = []
        for j in range(64):
            px = image[int(i + (j / 2))]
            if j % 2 == 0:
                px = px & 0xF
            else:
                px = int(px / 0x10)
            tile.append(px)
        tiles.append(tile)
    debug("Total number of tiles read: {}".format(len(tiles)))

    offset = read_pointer(bytes, tileset_pointer + 8)
    debug("Palette pointer: {:#x}".format(offset))
    if GAME in ["fr", "lg"]:
        palette_range = range(7) if primary == 0 else range(7, 16)
    else:
        palette_range = range(6) if primary == 0 else range(6, 16)
    palettes = []
    for i in palette_range:
        palette = []
        for j in range(16):
            colours = struct.unpack(
                "<H",
                bytes[
                    (offset + (i * 32) + (j * 2)) : (offset + (i * 32) + ((j + 1) * 2))
                ],
            )[0]
            (r, g, b) = (colours & 0x1F, (colours >> 5) & 0x1F, colours >> 10)
            (r, g, b) = (r * 8, g * 8, b * 8)
            palette.append((r, g, b))
        palettes.append(palette)

    offset = read_pointer(bytes, tileset_pointer + 12)
    end = read_pointer(bytes, tileset_pointer + 20)
    if GAME in ["em", "ru", "sa"]:
        end = read_pointer(bytes, tileset_pointer + 16)
    total_blocks = (end - offset) / 16
    debug("trying to read {} blocks".format(total_blocks))
    blocks = []
    for i in range(int(total_blocks)):
        blocks.append(read_block(bytes, offset, i))

    return (palettes, tiles, blocks)


def read_second_blocks(bytes, header_pointer):
    map_pointer = read_pointer(bytes, header_pointer)
    tileset_pointer = read_pointer(bytes, map_pointer + 20)
    offset = read_pointer(bytes, tileset_pointer + 12)
    blocks = []
    for i in range(96):
        b = read_block(bytes, offset, i)
        blocks.append(b)
    return blocks


def read_block(bytes, offset, i):
    block = []
    for j in range(8):
        block_data = struct.unpack(
            "<H", bytes[(offset + i * 16 + j * 2) : (offset + i * 16 + (j + 1) * 2)]
        )[0]
        palette = block_data >> 12
        tile = block_data & 0x3FF
        attributes = (block_data >> 10) & 0x3
        block.append((palette, tile, attributes))
    return block


def draw_block(map_pixels, palettes, tiles, blocks, x, y, block_num):
    # The first four tiles are the bottom tiles and the last four are the top
    # ones. The top tiles also have a mask to them, so we have to draw them
    # differently.
    block = blocks[block_num]
    for i, (palette, tile, attributes) in enumerate(block):
        x_offset = (i % 2) * 8
        y_offset = int((i % 4) / 2) * 8
        if tile < len(tiles) and palette < len(palettes):
            draw_tile(
                map_pixels,
                palettes[palette],
                tiles[tile],
                x + x_offset,
                y + y_offset,
                attributes,
                i >= 4,
            )
        else:
            print(f"TileError: Tile {tile}/{len(tiles)}, Palette {palette}/{12}")


def draw_tile(map_pixels, palette, tile, x, y, attributes, mask_mode):
    x_flip = attributes & 0x1
    y_flip = attributes & 0x2
    for i, px in enumerate(tile):
        x_offset = i % 8
        if x_flip:
            x_offset = 8 - (x_offset + 1)

        y_offset = int(i / 8)
        if y_flip:
            y_offset = 8 - (y_offset + 1)

        if mask_mode and px == 0:
            continue
        colour = palette[px]
        map_pixels[y + y_offset, x + x_offset] = colour + (255,)


def draw_map(bytes, map_, print_info=None):
    multi = True if print_info is not None else False
    if multi:
        m, b = print_info
        print(f"Drawing Map {m}.{b}{' '*5}")

    if read_pointer(bytes, map_) < 0:
        return None
    (width, height, label, tile_sprites, global_pointer, local_pointer) = read_map(
        bytes, map_
    )

    if None in [width, height, label, tile_sprites, global_pointer, local_pointer]:
        return None

    if global_pointer < 0 or local_pointer < 0:
        return None

    map_pixels = np.full((height * 16, width * 16, 4), [0, 0, 0, 0], dtype="uint8")

    (palettes, tiles, blocks) = read_tileset(bytes, global_pointer)
    if None not in [palettes, tiles, blocks]:
        (extra_palettes, extra_tiles, extra_blocks) = read_tileset(bytes, local_pointer)
        if None not in [extra_palettes, extra_tiles, extra_blocks]:
            palettes.extend(extra_palettes)
            tiles.extend(extra_tiles)
            blocks.extend(extra_blocks)

        block_error = False
        for (x, y) in tile_sprites:
            if tile_sprites[(x, y)] < len(blocks):
                draw_block(
                    map_pixels,
                    palettes,
                    tiles,
                    blocks,
                    0 + x * 16,
                    0 + y * 16,
                    tile_sprites[(x, y)],
                )
            else:
                if not block_error:
                    #print("  -BlockError")
                    block_error = True

    return map_pixels


# Returns a set of maps and (x, y) coordinates that the maps should be drawn at.
# The (x, y) coordinates are specified in blocks, not pixels.
# Coordinates originate from the top-left of a map.
def calculate_map_offsets(maps, bank_num, map_num):
    visited = {}

    def calculate_helper(map_id, caller_id, caller_offset, direction, offset):
        if map_id in visited:
            return []
        (bank, map_) = map_id
        (c_bank, c_map) = caller_id
        width = maps[bank][map_]["width"]
        height = maps[bank][map_]["height"]
        if width > 16000 or height > 16000:
            return []
        c_width = maps[c_bank][c_map]["width"]
        c_height = maps[c_bank][c_map]["height"]
        visited[map_id] = True

        (caller_x, caller_y) = caller_offset
        if direction == 0x1:  # Down
            coord = (caller_x + offset, caller_y + c_height)
        elif direction == 0x2:  # Up
            coord = (caller_x + offset, caller_y - height)
        elif direction == 0x3:  # Left
            coord = (caller_x - width, caller_y + offset)
        elif direction == 0x4:  # Right
            coord = (caller_x + c_width, caller_y + offset)
        elif direction == 0xF:  # Special starting value
            coord = (caller_x, caller_y)
        else:
            return []

        coords = [(map_id, coord)]
        for c in maps[bank][map_]["connections"]:
            try:
                coords.extend(
                    calculate_helper(
                        (c["map_bank"], c["map_number"]),
                        map_id,
                        coord,
                        c["direction"],
                        c["offset"],
                    )
                )
            except Exception as e:
                print(e)

        return coords

    return calculate_helper((bank_num, map_num), (bank_num, map_num), (0, 0), 0xF, 0)


def is_pointer(bytes, offset):
    if len(str(offset)) >= 8:
        return False
    return read_pointer(bytes, offset) > 0


def read_pointer(bytes, offset):
    load_address = int("0x8000000", 16)
    return read_int(bytes, offset) - load_address


def read_uint(bytes, offset):
    if len(str(offset)) >= 8:
        return 0
    if isinstance(offset, str):
        offset = int(offset, 16)
    return struct.unpack(b"<i", bytes[offset : (offset + 4)])[0]


def read_int(bytes, offset):
    if len(str(offset)) >= 8:
        return 0
    return struct.unpack(b"<I", bytes[offset : (offset + 4)])[0]


def CheckExt(choices):
    class Act(argparse.Action):
        def __call__(self, parser, namespace, fname, option_string=None):
            ext = os.path.splitext(fname)[1][1:]
            if ext not in choices:
                option_string = "({})".format(option_string) if option_string else ""
                parser.error(
                    "file doesn't end with one of {}{}".format(choices, option_string)
                )
            else:
                setattr(namespace, self.dest, fname)

    return Act


if __name__ == "__main__":
    start = time.time()
    main()
    print("Done in", round(time.time() - start, 2), "seconds")
