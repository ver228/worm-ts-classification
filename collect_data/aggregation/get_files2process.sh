python "$HOME/Github/tierpsy-tracker/cmd_scripts/processMultipleFiles.py" --mask_dir_root "$HOME/workspace/WormData/screenings/Serena_WT_Screening/MaskedVideos/" \
 --tmp_dir_root "" --only_summary --json_file _AEX_RIG.json --is_debug | tee "$HOME/workspace/tierpsy_output.txt"
python filter_files2process.py
 cp ~/workspace/files2process.txt ./
