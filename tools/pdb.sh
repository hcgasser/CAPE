#!/bin/bash

pdb()
{
  compgen -A function -abek pdb.
}

pdb.AF.from_fasta()
{
  if [ $# -ne 1 ]; then
    echo "Usage: fasta_to_pdb <fasta filename>"
  elif ! [ -f $1 ]; then
    echo "$1 needs to be a file"
  else
    datei=$1
    pfad=${datei%/*}
    filename=${datei##*/}
    seq=${filename%.*}
    ext=${filename##*.}

    pdb_output_file="${pfad}/${seq}/${seq}_AF.pdb"
    if [ -f "$datei" ] && [ "$ext" == "fasta" ] && ! [ -f "$pdb_output_file" ]; then
      echo "--------------------------------------------------------------------------------"
      echo "--------------------------------------------------------------------------------"
      echo "--------------------------------------------------------------------------------"
      echo "Run alphafold for "
      echo "  seq: ${seq}"
      echo "  in:  ${pfad}"
      echo "--------------------------------------------------------------------------------"

      if [ -z "$ALPHAFOLD_DB_PRESET" ]; then  
         ALPHAFOLD_DB_PRESET='full_dbs'   # reduced_dbs
      fi
      
      echo "Run with: ${ALPHAFOLD_DB_PRESET} - ${ALPHAFOLD_DATA}"

      python3 ${ALPHAFOLD_REPO}/docker/run_docker.py \
        --fasta_paths=${datei} \
        --max_template_date=2030-01-01 \
        --data_dir=$ALPHAFOLD_DATA \
        --output_dir=$pfad \
        --db_preset=$ALPHAFOLD_DB_PRESET

      # delete the unnecessary alphafold info
      rm -rf ${pfad}/${seq}/msas
      for f in features.pkl ranked_1.pdb ranked_2.pdb ranked_3.pdb ranked_4.pdb ranking_debug.json relax_metrics.json timings.json; do
        rm ${pfad}/${seq}/${f}
      done
      mv ${pfad}/${seq}/ranked_0.pdb ${pdb_output_file}

      for j in 1 2 3 4 5; do
        if [ ! -f ${pfad}/${seq}/relaxed_model_${j}_pred_0.pdb ]; then
          rm ${pfad}/${seq}/result_model_${j}_pred_0.pkl
        else
          mv ${pfad}/${seq}/result_model_${j}_pred_0.pkl ${pfad}/${seq}/result_model.pkl
        fi
        rm ${pfad}/${seq}/unrelaxed_model_${j}_pred_0.pdb
      done
    fi
  fi
}

pdb.AF.from_hashes()
{
    if [ $# -ne 2 ] && [ $# -ne 3 ]; then
        echo "Usage: pdb.AF.from_hashes <hash file> <protein directory> <optional: percentage of hashes to convert as an integer number>"
    else
        threshold=200
    	if [ $# -eq 3 ]; then
    	    threshold=${3}
    	fi
    	    
        while IFS= read -r line; do
            seq_hash=$line
            random_variable=$(( ($RANDOM % 100) ))

            if [ "$seq_hash" != "" ]; then
              if (( $random_variable < $threshold )); then
                if ! [ -e ${2}/structures/AF/pdb/${seq_hash}_AF.pdb ]; then
                  temp_dir=$(mktemp -d)
                  cp ${2}/sequences/${seq_hash}.fasta ${temp_dir}
                  pdb.AF.from_fasta ${temp_dir}/${seq_hash}.fasta
                  pdb_file=${temp_dir}/${seq_hash}/${seq_hash}_AF.pdb
                  pkl_file=${temp_dir}/${seq_hash}/result_model.pkl
                  if [ -e "${pdb_file}" ] && [ -e "${pkl_file}" ]; then
                      mkdir -p ${2}/structures/AF/pdb
                      mkdir -p ${2}/structures/AF/pkl
                      mv ${pdb_file} ${2}/structures/AF/pdb/${seq_hash}_AF.pdb
                      mv ${pkl_file} ${2}/structures/AF/pkl/${seq_hash}.pkl
                  else
                        ls -al "${temp_dir}"
                      echo "ERROR: ${pdb_file} or ${pkl_file} do not exist"
                  fi
                  rm -rf "${temp_dir}"
                fi
              fi
            fi
        done < "$1"
    fi
}
