for i in 01 02 03 04 05 06 07 08 09 10 11 12
do	
	DIR=jsicking@dpl"$i".kdlan.iais.fraunhofer.de:/data/user/jsicking/vwplatform/experiments/data/cifar/
	echo "$DIR"	
	scp -r /home/IAIS/jsicking/vw_collaborative_learning/noniid_experiments/index_lists/ $DIR
done
