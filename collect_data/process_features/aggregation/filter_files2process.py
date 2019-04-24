#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:43:56 2018

@author: avelinojaver
"""

from pathlib import Path
import sys


dflt_src_file = Path.home() / 'workspace' / 'tierpsy_output.txt'
dflt_src_file = Path.home() / 'workspace' / 'files2process.txt'

if __name__ == '__main__':
	import pdb
	pdb.set_trace()

	if len(sys.argv) == 1:



	

	split_token = '\n*********************************************\n'

	with open(src_file) as fid:
	    out = fid.read()
	#%%
	parts = out.split(split_token)
	cmds = ['python ' +  x.partition(' ')[-1] for x in parts[4].split('\n')]

	with open(dst_file, 'w') as fid:
	    fid.write('\n'.join(cmds))