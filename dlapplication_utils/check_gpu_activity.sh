for i in "$@"
do	
	echo ""
	echo dpl"$i"
	echo ""
	ssh dpl"$i".kdlan.iais.fraunhofer.de nvidia-smi
done
