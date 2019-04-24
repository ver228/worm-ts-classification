#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:43:56 2018

@author: avelinojaver
"""

from pathlib import Path
import sys


dflt_src_file = Path.home() / 'workspace' / 'tierpsy_output.txt'
dflt_dst_file = Path.home() / 'workspace' / 'files2process.txt'

if __name__ == '__main__':
	if len(sys.argv) == 1:
		src_file, dst_file = dflt_src_file, dflt_dst_file
	elif len(sys.argv) == 2:
		src_file, dst_file = sys.argv[1], dflt_dst_file
	else:
		src_file, dst_file = sys.argv[1:3]

	print(src_file, dst_file)

	split_token = '\n*********************************************\n'

	with open(src_file) as fid:
	    out = fid.read()
	#%%
	parts = out.split(split_token)
	cmds = ['python ' +  x.partition(' ')[-1] for x in parts[4].split('\n')]

	with open(dst_file, 'w') as fid:
	    fid.write('\n'.join(cmds))