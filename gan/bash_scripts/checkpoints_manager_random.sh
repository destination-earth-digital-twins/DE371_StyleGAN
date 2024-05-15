#!usr/bin/env bash


######### This file is a bash script to monitor experiment stopping due to time limit on LabIA infra
######### it should automatically relaunch an experiment which has been stopped by time out
######### by submitting a new job with network initialized to the last available checkpoint
######### if the experiment has terminated normally, nothing happens and the manager will
######### terminate after a laspe of inactivity

KEYSENTENCE="DUE TO TIME LIMIT"

STOPSENTENCE="RuntimeError: Horovod detected"

DATA_DIR=$1

SLURM_FILE=$2

EXPE_SET=$3

IS_LAUNCHED="false"

SLEEP_TIME=1800

MAX_COUNT=$4 #number of times we will wait for timeout and relaunch

TIME_OUT=$5 # expected time out

MAX_TIME=$(( $TIME_OUT + 60 )) 

echo "Starting to check for $SLURM_FILE";

while [[ ! -f "$SLURM_FILE" ]]; do

    echo "Script $SLURM_FILE still pending"
    sleep 60
    
done

echo "Starting to watch for termination on $SLURM_FILE";

timeout $MAX_TIME tail -f $SLURM_FILE | while read ;

	do

	sleep $SLEEP_TIME
	
	ABORT=$(grep -F -s $STOPSENTENCE $SLURM_FILE)

	IS_STOPPED=$(grep -F -s $KEYSENTENCE $SLURM_FILE)
        
        if [[ (${#ABORT} -gt 0) ]];
              then 
              
              echo "Stopping there because job aborted"
              
              sleep $MAX_TIME
        fi
	
	if [[ (${#IS_STOPPED} -gt 0) ]] ;
	  then
	  	CKPT=$(ls "$DATA_DIR/models/" | sed -e s/[^0-9]//g | sort -n | tail -1) ;
	  	echo  "Checkpoint at $CKPT";
	  	echo "Changing rights to be clean";
	  	bash /home/mrmn/poulainauzeaul/stylegan4arome/cleaner.sh >/dev/null 2>&1 ;
	  	echo "Launching new exp";
	  	SLURM_FILE="$(python3 /home/mrmn/poulainauzeaul/stylegan4arome/gan_horovod/expe_init_random.py --pretrained_model=$CKPT --SET_NUM=$EXPE_SET --max_relaunch=$(($MAX_COUNT - 1 )) | tail -1)" ;

          
	  else
	   
	   echo "Slurm job has not terminated, retrying in $SLEEP_TIME seconds"
	   
	fi
	
	
	done
echo "Stopping there due to time out"
