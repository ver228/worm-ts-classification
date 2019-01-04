SCREENING_ROOT="$HOME/workspace/WormData/screenings"
SAVE_NAME='allfiles2process.txt'
> $SAVE_NAME

TMP_TIERPSY_OUT="tierpsy_output.txt"
TMP_FILTERED="files2process.txt"


for s in `cat screens_list.txt` ; 
do
	echo $s
	#tierpsy_process --mask_dir_root $MASK_DIR --tmp_dir_root "" --only_summary --json_file _AEX_RIG.json --is_debug 

done

#tierpsy_process --mask_dir_root "$HOME/workspace/WormData/screenings/Serena_WT_Screening/MaskedVideos/" \
# --tmp_dir_root "" --only_summary --json_file _AEX_RIG.json --is_debug | tee "$HOME/workspace/tierpsy_output.txt"
#python filter_files2process.py
#cp ~/workspace/files2process.txt ./
