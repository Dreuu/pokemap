A version of pokemap using PIL since pygame is finicky about installing.

### Usage: 
`python pokemap.py ROM_FILENAME [options...]`

### Options:

`-g`: (Found automatically if not defined) Specify which game the ROM contains. Options are 'fr', 'lg', 'em', 'sa', or 'ru'  
`-t`: Number of threads to use for processing  
`-d`: Which maps to draw. Options are 'all' for all maps, and 'group' for only map groups. If given any other value, only the largest map group will be drawn (Default)  
`-o`: Specify the output file naming scheme and extension. Ex: 'fr_map.png'  
#

nlzss taken from https://github.com/magical/nlzss

article [here](https://medium.com/@mmmulani/creating-a-game-size-world-map-of-pok%C3%A9mon-fire-red-614da729476a#.1ruzb9nwl)
