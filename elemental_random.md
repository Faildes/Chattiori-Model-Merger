***Original Documented by hako-mikan***
# Elemental Random Merge
- This is a block-by-block merge that goes beyond block-by-block merge.

In a block-by-block merge, the merge ratio can be changed for each of the 25 blocks, but a blocks also consists of multiple elements, and in principle it is possible to change the ratio for each element. It is possible, but the number of elements is more than 600, and it was doubtful whether it could be handled by human hands, but we tried to implement it. I do not recommend merging elements by element out of the blue. It is recommended to use it as a final adjustment when a problem that cannot be solved by block-by-block merging.  

## Usage
Note that elemental merging is effective for both normal and block-by-block merging, and is computed last, so it will overwrite values specified for block-by-block merging.

Set in Elemental Merge. Note that if text is set here, it will be automatically adapted. Each element is listed in the table below, but it is not necessary to enter the full name of each element.  
You can check to see if the effect is properly applied by activating "print change" check. If this check is enabled, the applied elements will be displayed on the command prompt screen during the merge.  

### Format
Bloks:Element:Ratio, Bloks:Element:Ratio,...  
or  
Bloks:Element:Ratio  
Bloks:Element:Ratio  
Bloks:Element:Ratio  

Multiple specifications can be specified by separating them with commas or newlines. Commas and newlines may be mixed.
Bloks can be specified in uppercase from BASE,IN00-M00-OUT11. If left blank, all Bloks will be applied. Multiple Bloks can be specified by separating them with a space.
Similarly, multiple elements can be specified by separating them with a space.  
Partial matching is used, so for example, typing "attn" will change both attn1 and attn2, and typing "attn2" will change only attn2. If you want to specify more details, enter "attn2.to_out" and so on.  

OUT03 OUT04 OUT05:attn2 attn1.to_out:0.5(0.3)  

the ratio of elements containing attn2 and attn1.to_out in the OUT03, OUT04 and OUT05 layers will be (generated ratio) * (1 - 0.3) + 0.5 * 0.3.
If the element column is left blank, all elements in the specified Blocks will change, and the effect will be the same as a block-by-block merge.
If there are duplicate specifications, the one entered later takes precedence. 

OUT06:attn:0.5,OUT06:attn2.to_k:0.2  

is entered, attn other than attn2.to_k in the OUT06 layer will be 0.5, and only attn2.to_k will be 0.2.  

You can invert the effect by first entering NOT.
This can be set by Blocks and Element.  

NOT OUT04:attn:1  

will set the ratio 1 to the attn of all Blocks except the OUT04 layer.  

OUT05:NOT attn proj:0.2  

will set all Blocks except attn and proj in the OUT05 layer to 0.2.

OUT05:ALL:0.2  

will set all Blocks in the OUT05 layer to 0.2.

## XY plot
Several XY plots for elemental merge are available.  
#### elemental
Creates XY plots for multiple elemental merges. Elements should be separated from each other by blank lines.

#### pinpoint element
Creates an XY plot with different values for a specific element. Do the same with elements as with Pinpoint Blocks, but specify alpha for the opposite axis. Separate elements with a new line or comma.  

### List of elements
Basically, it seems that attn is responsible for the face and clothing information. The IN07, OUT03, OUT04, and OUT05 layers seem to have a particularly strong influence. It does not seem to make sense to change the same element in multiple Blocks at the same time, since the degree of influence often differs depending on the Blocks.
No element exists where it is marked null.

||IN00|IN01|IN02|IN03|IN04|IN05|IN06|IN07|IN08|IN09|IN10|IN11|M00|M00|OUT00|OUT01|OUT02|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|OUT09|OUT10|OUT11
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
op.bias|null|null|null||null|null||null|null||null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null
op.weight|null|null|null||null|null||null|null||null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null
emb_layers.1.bias|null|||null|||null|||null|null|||||||||||||||
emb_layers.1.weight|null|||null|||null|||null|null|||||||||||||||
in_layers.0.bias|null|||null|||null|||null|null|||||||||||||||
in_layers.0.weight|null|||null|||null|||null|null|||||||||||||||
in_layers.2.bias|null|||null|||null|||null|null|||||||||||||||
in_layers.2.weight|null|||null|||null|||null|null|||||||||||||||
out_layers.0.bias|null|||null|||null|||null|null|||||||||||||||
out_layers.0.weight|null|||null|||null|||null|null|||||||||||||||
out_layers.3.bias|null|||null|||null|||null|null|||||||||||||||
out_layers.3.weight|null|||null|||null|||null|null|||||||||||||||
skip_connection.bias|null|null|null|null||null|null||null|null|null|null|null|null||||||||||||
skip_connection.weight|null|null|null|null||null|null||null|null|null|null|null|null||||||||||||
norm.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
norm.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
proj_in.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
proj_in.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
proj_out.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
proj_out.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn1.to_k.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn1.to_out.0.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn1.to_out.0.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn1.to_q.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn1.to_v.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn2.to_k.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn2.to_out.0.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn2.to_out.0.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn2.to_q.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.attn2.to_v.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.ff.net.0.proj.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.ff.net.0.proj.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.ff.net.2.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.ff.net.2.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.norm1.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.norm1.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.norm2.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.norm2.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.norm3.bias|null|||null|||null|||null|null|null||null|null|null|null|||||||||
transformer_blocks.0.norm3.weight|null|||null|||null|||null|null|null||null|null|null|null|||||||||
conv.bias|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null||null|null||null|null||null|null|null
conv.weight|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null||null|null||null|null||null|null|null
0.bias||null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|
0.weight||null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|
2.bias|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|
2.weight|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|
time_embed.0.weight||null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|
time_embed.0.bias||null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|
time_embed.2.weight||null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|
time_embed.2.bias||null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|null|
